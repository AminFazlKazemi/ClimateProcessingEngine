#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py – نسخه نهایی سریال (بدون هیچ موازی‌سازی)
================================================================================
- اسکن Zarr از ابتدا (بدون checkpoint)
- پردازش سریال متغیرها در هر بلوک (یک به یک)
- نوشتن فقط متغیرهای موجود در block_result
- آزمون گرابز فعال
- بدون Dask، بدون ThreadPool، بدون هیچ parallelism
================================================================================
"""

import os
import sys
import gc
import time
import yaml
import numpy as np
import xarray as xr
import logging
from tqdm import tqdm

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
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger
from data_adapter import create_adapter

# ============================================================================
# خاموش کردن لاگ‌های مزاحم کش
# ============================================================================
for name in logging.root.manager.loggerDict:
    if any(x in name for x in ['data_adapter', 'io_pipeline', 'cache']):
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).disabled = True
logging.getLogger().setLevel(logging.INFO)

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
# چک کردن وضعیت یک بلوک از Zarr
# ============================================================================
def check_block_status(ds, block_idx, vars_list, block_size):
    start = block_idx * block_size
    end = min(start + block_size, ds.sizes['point'])
    
    block_missing = {}
    for var in vars_list:
        mean_var = f'{var}_mean'
        if mean_var not in ds:
            block_missing[var] = list(range(end - start))
            continue
        
        data = ds[mean_var].isel(day=slice(None)).values[:, start:end]
        is_missing = np.isnan(data).any(axis=0)
        missing_indices = np.where(is_missing)[0].tolist()
        
        if missing_indices:
            block_missing[var] = missing_indices
    
    return block_missing

# ============================================================================
# تابع اصلی
# ============================================================================
def main():
    vars_to_process = ['tmin', 'tmean', 'tmax']
    
    logger.info("=" * 80)
    logger.info("🚀 CLIMATOLOGY ENGINE – FINAL SERIAL VERSION (NO PARALLEL)")
    logger.info(f"   Variables: {vars_to_process}")
    logger.info(f"   Grubbs test: enabled (alpha=0.05, max_iter=3)")
    logger.info("   🔄 Serial mode: one variable at a time (fastest for this workload)")
    logger.info("   💾 Existing data preserved (only missing stations are filled)")
    logger.info("=" * 80)
    
    warmup_numba()
    
    # ============================================================
    # ساخت Adapter و جداول
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

    doy_table, _ = build_doy_table_from_config()
    tables = build_runtime_tables(ZARR_BASE)
    window_table = tables["window_table"]
    logger.info("   ✅ Tables built")

    # ============================================================
    # آماده‌سازی Zarr
    # ============================================================
    if not os.path.exists(OUTPUT_ZARR):
        logger.warning(f"⚠️ Output Zarr not found: {OUTPUT_ZARR}")
        logger.info("   Creating new Zarr store...")
        root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   📂 Zarr store created: {OUTPUT_ZARR}")
    else:
        root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   📂 Zarr store ready: {OUTPUT_ZARR}")
    
    total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
    logger.info(f"📦 Total blocks: {total_blocks}")

    # ============================================================
    # پردازش سریال (یک بلوک، یک متغیر)
    # ============================================================
    start_time = time.time()
    processed_blocks = 0
    incomplete_blocks_found = 0
    
    for block_idx in tqdm(range(total_blocks), desc="Processing blocks", unit="block"):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        
        # اسکن بلوک
        ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        block_missing = check_block_status(ds, block_idx, vars_to_process, BLOCK_SIZE)
        ds.close()
        
        if not block_missing:
            continue
        
        incomplete_blocks_found += 1
        missing_vars = [v for v in vars_to_process if v in block_missing and block_missing[v]]
        
        if not missing_vars:
            continue
        
        logger.info(f"   🔄 Block {block_idx}: {len(missing_vars)} variables missing: {missing_vars}")
        
        # پردازش سریال هر متغیر
        for var_name in missing_vars:
            var_idx = VARS.index(var_name)
            missing_indices = block_missing[var_name]
            
            logger.info(f"   🔄 Block {block_idx}, {var_name}: {len(missing_indices)} stations missing")
            
            try:
                process_block(
                    block_start=block_start,
                    block_end=block_end,
                    block_idx=block_idx,
                    file_map=adapter.file_map,
                    doy_table=doy_table,
                    window_table=window_table,
                    year_list=year_list,
                    root=root,
                    var_idx=var_idx,
                    last_checkpoint_station=0,
                    adapter=adapter,
                )
                logger.info(f"   ✅ Block {block_idx}, {var_name}: completed")
            except Exception as e:
                logger.error(f"   ❌ Error processing {var_name} in block {block_idx}: {e}")
                raise
        
        processed_blocks += 1
        gc.collect()
        
        if processed_blocks % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(f"   📊 Processed {processed_blocks} incomplete blocks, elapsed: {elapsed/60:.1f} min")
    
    # ============================================================
    # نهایی‌سازی
    # ============================================================
    logger.info("Finalizing Zarr...")
    ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
    ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
    ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}"
    ds.attrs["version"] = "12.0-serial-final"
    ds.attrs["outlier_detection"] = "Grubbs test (alpha=0.05, max_iter=3)"
    ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
    ds.close()

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"✅ PROCESSING COMPLETED SUCCESSFULLY in {total_time/60:.1f} minutes")
    logger.info(f"   Processed {processed_blocks} incomplete blocks")
    logger.info(f"   Total incomplete blocks found: {incomplete_blocks_found}")
    logger.info(f"   Variables: {vars_to_process}")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()