#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tmin_tmax_safe.py - پردازش فقط tmin و tmax از ابتدا با ریست checkpoint
"""

import os
import sys
import time
import yaml
import gc
import numpy as np
import xarray as xr
import logging

sys.path.insert(0, os.path.dirname(__file__))

# بارگذاری تنظیمات
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
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger
from checkpoint_manager import reset_checkpoint, save_checkpoint, delete_checkpoint
from data_adapter import create_adapter

# تنظیمات Dask (اختیاری)
dask_client = None
if USE_PARALLEL:
    try:
        from dask.distributed import Client
        dask_client = Client(processes=False, threads_per_worker=CORES)
        logger.info("   🚀 Dask Client started")
    except Exception as e:
        logger.warning(f"   ⚠️ Dask failed: {e}")

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

def main():
    # ریست checkpoint به بلوک ۰ (شروع از ابتدا)
    reset_checkpoint(0, 0)
    logger.info("🔄 Checkpoint reset to block 0, station 0")

    warmup_numba()
    logger.setLevel(logging.INFO)
    
    logger.info("=" * 80)
    logger.info("🚀 SAFE TMIN & TMAX PROCESSING (from scratch)")
    logger.info(f"   Years: {YEAR_START}–{YEAR_END} ({N_YEARS} years)")
    logger.info(f"   Days: {N_DAYS}")
    logger.info(f"   Variables: ['tmin', 'tmax']")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info(f"   Block Size: {BLOCK_SIZE}")
    logger.info("=" * 80)

    try:
        # ساخت adapter
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

        # جداول
        logger.info("Building tables...")
        doy_table, _ = build_doy_table_from_config()
        tables = build_runtime_tables(ZARR_BASE)
        window_table = tables["window_table"]
        logger.info("   ✅ Tables built")

        # Zarr store (اگر وجود داشته باشد باز می‌شود، در غیر این صورت ایجاد می‌شود)
        root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   📂 Zarr store ready: {OUTPUT_ZARR}")

        # شروع از بلوک ۰
        start_block = 0
        start_station = 0
        logger.info(f"   ⏩ Starting from block {start_block}, station {start_station}")

        total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
        logger.info(f"   Total blocks: {total_blocks}")

        # فقط tmin (0) و tmax (2)
        var_indices = [0, 2]

        for block_idx in range(start_block, total_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, n_stations)

            # پردازش tmin و tmax به ترتیب
            for var_idx in var_indices:
                var_name = VARS[var_idx]
                logger.info(f"   🔄 Processing {var_name} for block {block_idx} (stations {block_start}-{block_end-1})")

                try:
                    result = process_block(
                        block_start=block_start,
                        block_end=block_end,
                        block_idx=block_idx,
                        file_map=adapter.file_map,
                        doy_table=doy_table,
                        window_table=window_table,
                        year_list=year_list,
                        root=root,
                        var_idx=var_idx,
                        last_checkpoint_station=0,  # از ابتدا
                        adapter=adapter,
                    )
                    save_checkpoint(block_idx, block_end - 1)
                    logger.info(f"   ✅ {var_name} block {block_idx} completed, checkpoint saved.")
                except Exception as e:
                    logger.error(f"   ❌ Error processing {var_name} in block {block_idx}: {e}")
                    save_checkpoint(block_idx, block_start)
                    raise

                gc.collect()

            # بعد از اتمام هر دو متغیر برای یک بلوک
            save_checkpoint(block_idx, block_end - 1)
            logger.info(f"   ✅ Block {block_idx} fully completed (tmin + tmax)")

        # نهایی‌سازی Zarr (اضافه کردن مختصات و متادیتا)
        logger.info("Finalizing Zarr...")
        ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}"
        ds.attrs["version"] = "4.0-tmin-tmax-safe"
        ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
        ds.close()

        delete_checkpoint()
        logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if dask_client:
            dask_client.close()

if __name__ == "__main__":
    main()