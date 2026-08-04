#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - نسخه نهایی با ذخیره‌سازی تدریجی و checkpoint هر ۱۰۰ ایستگاه
================================================================================
- پردازش ایستگاهی با ذخیره‌سازی هر ۱۰۰ ایستگاه در Zarr
- قابلیت ادامه از آخرین checkpoint (حتی در همان بلوک)
- بدون نیاز به checkpoint.csv خارجی (از خود Zarr و فایل checkpoint داخلی استفاده می‌کند)
- شناسایی خودکار بلوک‌های کامل و رد شدن از آنها
================================================================================
"""

import os
import sys
import gc
import yaml
import time
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm
from pathlib import Path

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
    USE_PARALLEL, CORES
)
from calendar_tables import build_doy_table_from_config
from runtime_tables import build_runtime_tables
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata, create_empty_block_result
from monitoring.logger import logger
from data_adapter import create_adapter
from numerical_engine.analyze_station import analyze_station
from numerical_engine.merge_results import merge_station_result
from result_pipeline.write_block import write_block_safe
from checkpoint_manager import save_checkpoint, load_checkpoint, delete_checkpoint

# ============================================================================
# Dask (اختیاری)
# ============================================================================
dask_client = None
if USE_PARALLEL:
    try:
        from dask.distributed import Client
        dask_client = Client(processes=False, threads_per_worker=CORES)
        logger.info("   🚀 Dask Client started")
    except Exception as e:
        logger.warning(f"   ⚠️ Dask failed: {e}")

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
# توابع تشخیص وضعیت
# ============================================================================

def get_progress(zarr_path, var_names, day_idx=0):
    """اسکن Zarr و تشخیص تعداد نقاط معتبر برای هر متغیر."""
    progress = {v: 0 for v in var_names}
    n_stations = 0

    if not os.path.exists(zarr_path):
        return progress, n_stations

    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        n_stations = ds.sizes.get('point', 0)

        for var in var_names:
            mean_var = f'{var}_mean'
            if mean_var in ds:
                data = ds[mean_var].isel(day=day_idx).values
                progress[var] = np.sum(~np.isnan(data))
            else:
                progress[var] = 0
        ds.close()
    except Exception as e:
        logger.warning(f"   ⚠️ Could not read Zarr: {e}")

    return progress, n_stations

def is_block_complete(ds, var_name, block_start, block_size, day_idx=0):
    """بررسی کامل بودن یک بلوک برای یک متغیر خاص."""
    mean_var = f'{var_name}_mean'
    if mean_var not in ds:
        return False
    try:
        data = ds[mean_var].isel(day=day_idx).values
        block_end = min(block_start + block_size, len(data))
        block_data = data[block_start:block_end]
        return np.all(~np.isnan(block_data))
    except Exception:
        return False

def find_start_block(zarr_path, var_names, block_size, day_idx=0):
    """پیدا کردن اولین بلوک ناقص."""
    progress, n_stations = get_progress(zarr_path, var_names, day_idx)

    if n_stations == 0:
        return 0, 0, 0

    total_blocks = (n_stations + block_size - 1) // block_size

    # اگر همه متغیرها کامل بودند
    if all(p >= n_stations for p in progress.values()):
        logger.info("🎉 همه متغیرها کامل هستند!")
        return total_blocks, n_stations, total_blocks

    # پیدا کردن اولین بلوک ناقص
    ds = None
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
    except:
        pass

    start_block = 0
    if ds is not None:
        for block_idx in range(total_blocks):
            block_start = block_idx * block_size
            all_complete = True
            for var in var_names:
                if not is_block_complete(ds, var, block_start, block_size, day_idx):
                    all_complete = False
                    break
            if not all_complete:
                start_block = block_idx
                break
        ds.close()

    logger.info(f"📌 اولین بلوک ناقص: {start_block}")
    logger.info(f"   پیشرفت: tmin={progress.get('tmin',0):,}, tmean={progress.get('tmean',0):,}, tmax={progress.get('tmax',0):,}")

    return start_block, n_stations, total_blocks

# ============================================================================
# توابع نوشتن تدریجی و مدیریت checkpoint
# ============================================================================

def write_partial_block(root, block_result, block_start, block_end, start_col, end_col):
    """
    نوشتن بخشی از block_result در Zarr (ستون‌های start_col تا end_col-1)
    """
    for name in block_result.keys():
        root[name][:, block_start + start_col : block_start + end_col] = block_result[name][:, start_col:end_col]

def get_checkpoint_key(block_idx, var_idx):
    """ساخت کلید checkpoint برای یک بلوک و متغیر خاص"""
    return f"checkpoint_block_{block_idx}_var_{var_idx}"

def save_incremental_checkpoint(block_idx, var_idx, station_idx):
    """ذخیره checkpoint برای یک بلوک و متغیر خاص"""
    cp = {
        "block": block_idx,
        "var_idx": var_idx,
        "station": station_idx,
        "timestamp": int(time.time())
    }
    # استفاده از checkpoint_manager برای ذخیره
    # اما checkpoint_manager فقط block و station کلی را ذخیره می‌کند.
    # برای ذخیره جزئی‌تر، از یک فایل جداگانه استفاده می‌کنیم.
    cp_file = Path(OUTPUT_DIR) / f"checkpoint_{block_idx}_{var_idx}.json"
    import json
    with open(cp_file, "w") as f:
        json.dump(cp, f)
    logger.debug(f"   💾 Checkpoint saved: block={block_idx}, var={var_idx}, station={station_idx}")

def load_incremental_checkpoint(block_idx, var_idx):
    """بارگذاری checkpoint برای یک بلوک و متغیر خاص"""
    cp_file = Path(OUTPUT_DIR) / f"checkpoint_{block_idx}_{var_idx}.json"
    if not cp_file.exists():
        return None
    import json
    try:
        with open(cp_file, "r") as f:
            cp = json.load(f)
        return cp
    except:
        return None

def delete_incremental_checkpoint(block_idx, var_idx):
    """حذف checkpoint برای یک بلوک و متغیر خاص"""
    cp_file = Path(OUTPUT_DIR) / f"checkpoint_{block_idx}_{var_idx}.json"
    if cp_file.exists():
        cp_file.unlink()

# ============================================================================
# تابع پردازش بلوک با ذخیره‌سازی تدریجی
# ============================================================================

def process_block_smart(block_start, block_end, block_idx, adapter,
                        doy_table, window_table, year_list, root,
                        var_indices_to_process):
    """
    پردازش یک بلوک با ذخیره‌سازی تدریجی هر ۱۰۰ ایستگاه.
    """
    block_size = block_end - block_start

    # ============================================================
    # ۱. خواندن داده‌ها (یک بار برای همه متغیرها)
    # ============================================================
    logger.info(f"   📂 Loading block {block_idx} (one time)...")

    data_dict = {}
    n_vars = len(VARS)
    total_files = sum(1 for year in year_list for month in range(1, 13)
                      if (year, month) in adapter.file_map)

    pbar_load = tqdm(total=total_files, desc="   Loading Zarr", unit="file", leave=False)

    for year in year_list:
        for month in range(1, 13):
            key = (year, month)
            if key not in adapter.file_map:
                continue

            if hasattr(adapter, 'load_block_all_vars'):
                combined_data = adapter.load_block_all_vars(
                    block_start=block_start,
                    block_size=block_size,
                    year_idx=year_list.index(year) if year in year_list else 0,
                    month=month
                )
                if combined_data is not None:
                    data_dict[key] = combined_data
            else:
                # Fallback: تک‌متغیره
                combined = None
                for v in range(n_vars):
                    var_data = adapter.load_block(
                        block_start=block_start,
                        block_size=block_size,
                        year_idx=year_list.index(year) if year in year_list else 0,
                        month=month,
                        var_idx=v
                    )
                    if var_data is not None:
                        if var_data.ndim == 2 and var_data.shape[0] == block_size:
                            var_data = var_data.T
                        elif var_data.ndim == 1:
                            var_data = var_data.reshape(-1, 1)
                        if combined is None:
                            days = var_data.shape[0]
                            combined = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
                        combined[:, :, v] = var_data
                if combined is not None:
                    data_dict[key] = combined
            pbar_load.update(1)
    pbar_load.close()

    if not data_dict:
        logger.warning(f"   ⚠️ No data loaded for block {block_idx}")
        return False

    # ============================================================
    # ۲. مونتاژ داده‌ها (یک بار)
    # ============================================================
    from io_pipeline.assemble_block import assemble_block
    block_data = assemble_block(data_dict, doy_table, block_size, year_list, var_idx=0)

    if hasattr(block_data, 'values'):
        block_data = block_data.values
    elif not isinstance(block_data, np.ndarray):
        block_data = np.array(block_data)

    if block_data.ndim != 4:
        logger.error(f"   ❌ Invalid block_data shape: {block_data.shape}")
        return False

    # ============================================================
    # ۳. پردازش ایستگاهی با ذخیره‌سازی تدریجی
    # ============================================================
    logger.info(f"   🔄 Processing {len(var_indices_to_process)} variables station-wise...")

    # برای هر متغیر، به صورت جداگانه پردازش می‌کنیم تا checkpoint مستقل داشته باشیم
    for var_idx in var_indices_to_process:
        var_name = VARS[var_idx]
        logger.info(f"      🔹 Processing {var_name}...")

        # بررسی checkpoint برای این متغیر
        cp = load_incremental_checkpoint(block_idx, var_idx)
        start_station_local = 0
        if cp is not None:
            start_station_local = cp.get("station", 0)
            if start_station_local >= block_size:
                # اگر checkpoint نشان می‌دهد که همه ایستگاه‌ها پردازش شده‌اند، رد کن
                logger.info(f"         ⏭️ {var_name} already fully processed in this block")
                continue
            logger.info(f"         ⏩ Resuming from station {start_station_local}")

        # ایجاد block_result با اندازه کامل
        block_result = create_empty_block_result(block_size)

        # اگر از وسط شروع می‌کنیم، باید داده‌های قبلی را از Zarr بخوانیم و در block_result قرار دهیم
        # ولی چون قرار است داده‌ها را به صورت تدریجی بنویسیم، نیازی به خواندن نیست.
        # فقط مطمئن می‌شویم که block_result برای بخش‌های قبلی خالی بماند (در نوشتن بعدی، آنها را نمی‌نویسیم).

        # Progress bar برای ایستگاه‌ها
        pbar_stations = tqdm(total=block_size - start_station_local,
                             desc=f"      {var_name} stations",
                             unit="station",
                             initial=start_station_local,
                             leave=True)

        # حلقه روی ایستگاه‌ها از start_station_local تا انتها
        for local_idx in range(start_station_local, block_size):
            station_idx = block_start + local_idx
            station_data = block_data[local_idx, :, :, :]

            # اطمینان از شکل صحیح
            if not isinstance(station_data, np.ndarray):
                station_data = np.array(station_data)
            if station_data.ndim == 0:
                station_data = station_data.reshape(1, 1, 1)
            elif station_data.ndim == 1:
                try:
                    station_data = station_data.reshape(len(year_list), N_DAYS, n_vars)
                except ValueError:
                    station_data = np.full((len(year_list), N_DAYS, n_vars), np.nan, dtype=np.float32)
            elif station_data.ndim == 2:
                if station_data.shape[0] == len(year_list) * N_DAYS and station_data.shape[1] == n_vars:
                    station_data = station_data.reshape(len(year_list), N_DAYS, n_vars)

            try:
                result = analyze_station(station_data, year_list, window_table, var_idx)
                if result is not None:
                    merge_station_result(block_result, result, local_idx)
            except Exception as e:
                logger.warning(f"      ⚠️ Station {station_idx}, var {var_name} failed: {e}")

            # به‌روزرسانی progress bar بعد از اتمام هر ایستگاه
            pbar_stations.update(1)

            # ============================================================
            # ذخیره‌سازی هر ۱۰۰ ایستگاه (یا در انتهای بلوک)
            # ============================================================
            if (local_idx + 1) % 100 == 0 or local_idx == block_size - 1:
                # نوشتن بخش پردازش‌شده در Zarr
                start_write = start_station_local
                end_write = local_idx + 1
                try:
                    write_partial_block(root, block_result, block_start, block_end, start_write, end_write)
                    logger.debug(f"      💾 Written stations {start_write}-{end_write-1} to Zarr")
                except Exception as e:
                    logger.error(f"      ❌ Write failed at station {local_idx}: {e}")
                    raise

                # ذخیره checkpoint
                save_incremental_checkpoint(block_idx, var_idx, local_idx + 1)

        pbar_stations.close()

        # بعد از اتمام همه ایستگاه‌ها، checkpoint را پاک می‌کنیم
        delete_incremental_checkpoint(block_idx, var_idx)
        logger.info(f"      ✅ {var_name} block {block_idx} completed and checkpoint removed")

        # آزادسازی حافظه برای این متغیر
        del block_result
        gc.collect()

    # آزادسازی حافظه
    del data_dict, block_data
    gc.collect()

    return True

# ============================================================================
# تابع اصلی
# ============================================================================

def main():
    logger.setLevel(logging.INFO)
    warmup_numba()

    logger.info("=" * 80)
    logger.info("🚀 SMART PROCESSING (Incremental write every 100 stations)")
    logger.info(f"   Years: {YEAR_START}–{YEAR_END} ({N_YEARS} years)")
    logger.info(f"   Days: {N_DAYS}")
    logger.info(f"   Variables: {VARS}")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info(f"   Block Size: {BLOCK_SIZE}")
    logger.info("=" * 80)

    try:
        # ============================================================
        # ۱. آماده‌سازی Adapter
        # ============================================================
        year_list = list(range(YEAR_START, YEAR_END + 1))

        DATA_FORMAT = CONFIG.get("data_format", "auto")
        N_POINTS_MAX = CONFIG.get("processing", {}).get("n_points_max", 40000)
        LAT_MIN = CONFIG.get("lat_min", None)
        LAT_MAX = CONFIG.get("lat_max", None)
        LON_MIN = CONFIG.get("lon_min", None)
        LON_MAX = CONFIG.get("lon_max", None)

        adapter = create_adapter(
            ZARR_BASE, year_list,
            data_format=DATA_FORMAT,
            cache_enabled=True,
            max_points=N_POINTS_MAX,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            lon_min=LON_MIN, lon_max=LON_MAX,
        )

        n_stations = adapter.n_points
        coords = adapter.get_coords()
        station_ids = coords.get("stationid", np.arange(n_stations))
        lons = coords["lon"]
        lats = coords["lat"]
        elevs = coords["elev"]

        logger.info(f"   ✅ {n_stations:,} stations loaded")

        # ============================================================
        # ۲. جداول
        # ============================================================
        logger.info("Building tables...")
        doy_table, _ = build_doy_table_from_config()
        tables = build_runtime_tables(ZARR_BASE)
        window_table = tables["window_table"]
        logger.info("   ✅ Tables built")

        # ============================================================
        # ۳. باز کردن Zarr و تشخیص نقطه شروع
        # ============================================================
        root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   📂 Zarr store ready: {OUTPUT_ZARR}")

        start_block, _, total_blocks = find_start_block(
            OUTPUT_ZARR, ['tmin', 'tmean', 'tmax'], BLOCK_SIZE, day_idx=0
        )

        if start_block >= total_blocks:
            logger.info("🎉 همه بلوک‌ها کامل هستند. پایان.")
            return

        logger.info(f"📦 شروع از بلوک {start_block} از {total_blocks}")

        # ============================================================
        # ۴. حلقه اصلی
        # ============================================================
        # باز کردن دیتاست برای بررسی بلوک‌ها
        ds = None
        if os.path.exists(OUTPUT_ZARR):
            try:
                ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
            except Exception:
                pass

        var_indices_all = [0, 1, 2]  # tmin, tmean, tmax

        for block_idx in range(start_block, total_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, n_stations)

            # ============================================================
            # تشخیص متغیرهای ناقص در این بلوک
            # ============================================================
            vars_to_process = []
            if ds is not None:
                for var_idx in var_indices_all:
                    var_name = VARS[var_idx]
                    # همچنین بررسی کنیم که آیا checkpoint جزئی برای این متغیر وجود دارد
                    cp = load_incremental_checkpoint(block_idx, var_idx)
                    if cp is not None and cp.get("station", 0) >= (block_end - block_start):
                        # اگر checkpoint نشان می‌دهد که کامل شده، رد کن
                        logger.info(f"   ⏭️ {var_name} block {block_idx} already complete (from checkpoint)")
                        continue
                    if not is_block_complete(ds, var_name, block_start, BLOCK_SIZE, day_idx=0):
                        vars_to_process.append(var_idx)
                    else:
                        logger.info(f"   ⏭️ {var_name} block {block_idx} is complete → skip")
            else:
                # اگر Zarr قابل خواندن نبود، همه را پردازش کن (با احتیاط)
                vars_to_process = var_indices_all.copy()

            if not vars_to_process:
                logger.info(f"   ✅ Block {block_idx} complete for all vars → skip")
                # به‌روزرسانی دیتاست برای ادامه
                if ds is not None:
                    ds.close()
                try:
                    ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
                except Exception:
                    ds = None
                continue

            logger.info(f"📦 Block {block_idx}: stations {block_start:,}-{block_end:,} "
                        f"(processing: {[VARS[v] for v in vars_to_process]})")

            # ============================================================
            # پردازش بلوک (با ذخیره‌سازی تدریجی)
            # ============================================================
            success = process_block_smart(
                block_start, block_end, block_idx, adapter,
                doy_table, window_table, year_list, root,
                vars_to_process
            )

            if not success:
                logger.error(f"   ❌ Block {block_idx} failed")
                break

            # به‌روزرسانی دیتاست (برای ادامه)
            if ds is not None:
                ds.close()
            try:
                ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
            except Exception:
                ds = None

            gc.collect()

        # ============================================================
        # ۵. نهایی‌سازی
        # ============================================================
        logger.info("Finalizing Zarr...")
        if ds is not None:
            ds.close()

        ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}"
        ds.attrs["version"] = "8.0-incremental-write"
        ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
        ds.close()

        logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if dask_client:
            dask_client.close()

if __name__ == "__main__":
    main()