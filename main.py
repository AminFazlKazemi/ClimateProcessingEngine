#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py – نسخه نهایی پایدار
================================================================================
- بدون چک‌پوینت
- همیشه از بلوک ۰ شروع می‌کند (بلوک‌های کامل را رد می‌کند)
- چانک‌های ۲۰۰ تایی با پردازش موازی (۴ کارگر)
- نوشتن هر چانک به‌محض پردازش
================================================================================
"""

import os
import sys
import gc
import yaml
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# بارگذاری تنظیمات
# ============================================================================
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

from constants import (
    YEAR_START, YEAR_END, N_YEARS, N_DAYS, VARS,
    OUTPUT_DIR, OUTPUT_ZARR, ZARR_BASE, BLOCK_SIZE,
)
from calendar_tables import build_doy_table_from_config
from runtime_tables import build_runtime_tables
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata, create_empty_block_result
from monitoring.logger import logger
from data_adapter import create_adapter
from numerical_engine.analyze_station import analyze_station
from numerical_engine.merge_results import merge_station_result

# ============================================================================
# تنظیمات سرعت
# ============================================================================
CHUNK_SIZE = 200
MAX_WORKERS = 4

# ============================================================================
# Warmup Numba
# ============================================================================
def warmup_numba():
    try:
        import numpy as np
        from numerical_engine.distributions import fit_distribution
        sample = np.random.randn(200).astype(np.float64)
        for _ in range(10):
            fit_distribution(sample)
        logger.info("   ✅ Numba JIT warmup completed")
    except Exception as e:
        logger.warning(f"   ⚠️ Warmup failed: {e}")

# ============================================================================
# توابع بررسی کامل بودن
# ============================================================================
def is_block_complete(ds, var_names, block_start, block_size, day_idx=0):
    """بررسی کامل بودن بلوک برای همه متغیرها (فقط با میانگین)"""
    for var_name in var_names:
        mean_var = f'{var_name}_mean'
        if mean_var not in ds:
            return False
        try:
            data = ds[mean_var].isel(day=day_idx).values
            block_end = min(block_start + block_size, len(data))
            block_data = data[block_start:block_end]
            if not np.all(~np.isnan(block_data)):
                return False
        except Exception:
            return False
    return True

def is_chunk_complete(ds, var_name, start, end, day_idx=0):
    mean_var = f'{var_name}_mean'
    if mean_var not in ds:
        return False
    try:
        data = ds[mean_var].isel(day=day_idx).values[start:end]
        return np.all(~np.isnan(data))
    except Exception:
        return False

def write_chunk(root, block_result, block_start, chunk_start):
    for name in block_result.keys():
        root[name][:, block_start + chunk_start : block_start + chunk_start + block_result[name].shape[1]] = block_result[name][:, :]

# ============================================================================
# پردازش یک چانک
# ============================================================================
def process_chunk(chunk_start, chunk_end, block_start, block_idx, adapter,
                  doy_table, window_table, year_list, root, vars_to_process, ds, block_data):
    chunk_size = chunk_end - chunk_start
    n_vars = len(VARS)

    block_result = create_empty_block_result(chunk_size)

    for local_idx in range(chunk_start, chunk_end):
        station_idx = block_start + local_idx
        station_data = block_data[local_idx, :, :, :]

        vars_here = []
        if ds is not None:
            for var_name in vars_to_process:
                mean_var = f'{var_name}_mean'
                if mean_var in ds:
                    val = ds[mean_var].isel(day=0, point=station_idx).values
                    if np.isnan(val):
                        vars_here.append(var_name)
        else:
            vars_here = vars_to_process.copy()

        if not vars_here:
            continue

        if not isinstance(station_data, np.ndarray):
            station_data = np.array(station_data)
        if station_data.ndim == 0:
            station_data = station_data.reshape(1, 1, 1)
        elif station_data.ndim == 1:
            try:
                station_data = station_data.reshape(len(year_list), N_DAYS, n_vars)
            except:
                station_data = np.full((len(year_list), N_DAYS, n_vars), np.nan, dtype=np.float32)
        elif station_data.ndim == 2:
            if station_data.shape[0] == len(year_list) * N_DAYS and station_data.shape[1] == n_vars:
                station_data = station_data.reshape(len(year_list), N_DAYS, n_vars)

        for var_name in vars_here:
            var_idx = VARS.index(var_name)
            try:
                result = analyze_station(station_data, year_list, window_table, var_idx)
                if result is not None:
                    merge_station_result(block_result, result, local_idx - chunk_start)
            except Exception as e:
                logger.warning(f"      ⚠️ Station {station_idx}, var {var_name} failed: {e}")

    write_chunk(root, block_result, block_start, chunk_start)
    del block_result
    gc.collect()

# ============================================================================
# تابع اصلی
# ============================================================================
def main():
    logger.setLevel(logging.INFO)
    warmup_numba()

    logger.info("=" * 80)
    logger.info(f"🚀 FINAL VERSION: CHUNK={CHUNK_SIZE}, WORKERS={MAX_WORKERS}, NO CHECKPOINT")
    logger.info(f"   Years: {YEAR_START}–{YEAR_END}")
    logger.info(f"   Variables: {VARS}")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info(f"   Block Size: {BLOCK_SIZE}")
    logger.info("=" * 80)

    year_list = list(range(YEAR_START, YEAR_END + 1))

    adapter = create_adapter(
        ZARR_BASE, year_list,
        data_format=CONFIG.get("data_format", "auto"),
        cache_enabled=True,
        max_points=CONFIG.get("processing", {}).get("n_points_max", 40000),
        lat_min=CONFIG.get("lat_min"), lat_max=CONFIG.get("lat_max"),
        lon_min=CONFIG.get("lon_min"), lon_max=CONFIG.get("lon_max"),
    )

    n_stations = adapter.n_points
    coords = adapter.get_coords()
    station_ids = coords.get("stationid", np.arange(n_stations))
    lons = coords["lon"]
    lats = coords["lat"]
    elevs = coords["elev"]

    logger.info(f"   ✅ {n_stations:,} stations loaded")

    doy_table, _ = build_doy_table_from_config()
    tables = build_runtime_tables(ZARR_BASE)
    window_table = tables["window_table"]
    logger.info("   ✅ Tables built")

    root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
    logger.info(f"   📂 Zarr store ready")

    total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
    logger.info(f"📦 تعداد کل بلوک‌ها: {total_blocks}")

    # ============================================================
    # باز کردن دیتاست برای بررسی کامل بودن
    # ============================================================
    ds = None
    if os.path.exists(OUTPUT_ZARR):
        try:
            ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        except Exception as e:
            logger.warning(f"   ⚠️ Could not open Zarr: {e}")

    var_names = VARS

    # ============================================================
    # حلقه اصلی از بلوک ۰ تا آخر
    # ============================================================
    for block_idx in range(0, total_blocks):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        block_size = block_end - block_start

        # ۱. چک کامل بودن بلوک
        if ds is not None and is_block_complete(ds, var_names, block_start, block_size, day_idx=0):
            logger.info(f"   ✅ Block {block_idx} is complete → skip")
            continue

        # ۲. بارگذاری داده
        logger.info(f"📂 Loading block {block_idx}...")
        data_dict = {}
        for year in year_list:
            for month in range(1, 13):
                key = (year, month)
                if key not in adapter.file_map:
                    continue
                combined = adapter.load_block_all_vars(
                    block_start=block_start,
                    block_size=block_size,
                    year_idx=year_list.index(year) if year in year_list else 0,
                    month=month
                )
                if combined is not None:
                    data_dict[key] = combined

        if not data_dict:
            logger.warning(f"   ⚠️ No data for block {block_idx}")
            continue

        from io_pipeline.assemble_block import assemble_block
        block_data = assemble_block(data_dict, doy_table, block_size, year_list, var_idx=0)
        if hasattr(block_data, 'values'):
            block_data = block_data.values
        elif not isinstance(block_data, np.ndarray):
            block_data = np.array(block_data)

        if block_data.ndim != 4:
            logger.error(f"   ❌ Invalid block_data shape: {block_data.shape}")
            continue

        # ۳. تشخیص چانک‌های ناقص
        chunks_to_process = []
        for chunk_start in range(0, block_size, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, block_size)

            vars_to_process = []
            if ds is not None:
                for var_name in var_names:
                    if not is_chunk_complete(ds, var_name, block_start + chunk_start, block_start + chunk_end, day_idx=0):
                        vars_to_process.append(var_name)
            else:
                vars_to_process = var_names.copy()

            if vars_to_process:
                chunks_to_process.append((chunk_start, chunk_end, vars_to_process))
            else:
                logger.info(f"   ✅ Block {block_idx}, chunk {chunk_start//CHUNK_SIZE+1}: complete → skip")

        if not chunks_to_process:
            logger.info(f"   ✅ Block {block_idx} all chunks complete → skip")
            continue

        logger.info(f"   🔄 Processing {len(chunks_to_process)} incomplete chunks with {MAX_WORKERS} workers...")

        # ۴. پردازش موازی چانک‌ها
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for chunk_start, chunk_end, vars_to_process in chunks_to_process:
                future = executor.submit(
                    process_chunk,
                    chunk_start, chunk_end, block_start, block_idx, adapter,
                    doy_table, window_table, year_list, root, vars_to_process, ds, block_data
                )
                futures.append((chunk_start, future))

            for chunk_start, future in futures:
                try:
                    future.result()
                    logger.info(f"   💾 Block {block_idx}, chunk {chunk_start//CHUNK_SIZE+1}: done")
                except Exception as e:
                    logger.error(f"   ❌ Block {block_idx}, chunk {chunk_start//CHUNK_SIZE+1} failed: {e}")

        # ۵. به‌روزرسانی دیتاست
        if ds is not None:
            ds.close()
        try:
            ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        except:
            ds = None

        del data_dict, block_data
        gc.collect()

    # ============================================================
    # نهایی‌سازی
    # ============================================================
    logger.info("Finalizing Zarr...")
    if ds is not None:
        ds.close()

    ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
    ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
    ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}"
    ds.attrs["version"] = "final-stable"
    ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
    ds.close()

    logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()