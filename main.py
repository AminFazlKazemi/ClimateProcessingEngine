#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - نقطه ورود اصلی با Data Adapter و Checkpoint صحیح
نسخه ۳.۳ - استفاده از checkpoint_manager با تشخیص خودکار
"""

import os
import sys
import time
import glob
import yaml
import gc
import numpy as np
import xarray as xr
import logging

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================================
# بارگذاری تنظیمات
# ============================================================================

CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

from constants import (
    YEAR_START,
    YEAR_END,
    N_YEARS,
    N_DAYS,
    VARS,
    N_VARS,
    VAR_INDEX_FOR_FIT,
    OUTPUT_DIR,
    OUTPUT_ZARR,
    ZARR_BASE,
    BLOCK_SIZE,
    USE_PARALLEL,
    CORES,
    VALIDATE_AFTER_LOAD,
    VALIDATE_BEFORE_WRITE,
    VALIDATE_EVERY_N_BLOCKS,
    FLOAT_DTYPE,
    INT_DTYPE
)

from calendar_tables import build_doy_table_from_config
from runtime_tables import build_runtime_tables
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger

# ============================================================================
# ✅ استفاده از checkpoint_manager با تشخیص خودکار
# ============================================================================
from checkpoint_manager import ensure_checkpoint, save_checkpoint, delete_checkpoint

# ============================================================================
# استفاده از Data Adapter
# ============================================================================

from data_adapter import create_adapter

DATA_FORMAT = CONFIG.get("data_format", "auto")
N_POINTS_MAX = CONFIG.get("processing", {}).get("n_points_max", 40000)
LAT_MIN = CONFIG.get("lat_min", None)
LAT_MAX = CONFIG.get("lat_max", None)
LON_MIN = CONFIG.get("lon_min", None)
LON_MAX = CONFIG.get("lon_max", None)

# تنظیمات تولید نقشه‌های صدک (از config.yaml خوانده می‌شود)
GENERATE_MAPS = CONFIG.get("generate_percentile_maps", True)
PERCENTILES = CONFIG.get("percentiles", [0.9, 0.95, 0.99])
MAP_DAYS = CONFIG.get("map_days", [1, 91, 181, 271])  # روزهای نمونه
MAP_OUTPUT_DIR = CONFIG.get("map_output_dir", os.path.join(OUTPUT_DIR, "percentile_maps"))

# ============================================================================
# تنظیمات Dask
# ============================================================================

dask_client = None
if USE_PARALLEL:
    try:
        from dask.distributed import Client
        dask_client = Client(processes=False, threads_per_worker=CORES)
        logger.info(f"   🚀 Dask Client started")
    except Exception as e:
        logger.warning(f"   ⚠️ Dask failed: {e}")

# ============================================================================
# تابع اصلی
# ============================================================================

def warmup_numba():
    """پیش‌کامپایل Numba قبل از شروع پردازش"""
    try:
        import numpy as np
        from numerical_engine.distributions import fit_distribution
        sample = np.random.randn(200).astype(np.float64)
        for _ in range(10):
            fit_distribution(sample)
        logger.info("   ✅ Numba JIT warmup completed")
    except Exception as e:
        logger.warning(f"   ⚠️ Warmup failed: {e}")


def generate_percentile_maps(zarr_path, output_dir, percentiles, days=None):
    """
    تولید نقشه‌های صدک از خروجی Zarr
    """
    try:
        from generate_percentile_maps import compute_percentile_map
        logger.info("🗺️ Generating percentile maps...")
        compute_percentile_map(
            zarr_path=zarr_path,
            percentiles=percentiles,
            output_dir=output_dir,
            days=days
        )
        logger.info("✅ Percentile maps generated successfully.")
    except ImportError as e:
        logger.warning(f"⚠️ Could not import generate_percentile_maps: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate percentile maps: {e}")


def main():
    # Warmup Numba (pre-compile JIT)
    warmup_numba()

    logger.setLevel(logging.INFO)
    
    logger.info("=" * 80)
    logger.info("🚀 CLIMATOLOGY PROCESSING ENGINE v3.3")
    logger.info(f"   Years: {YEAR_START}–{YEAR_END} ({N_YEARS} years)")
    logger.info(f"   Days: {N_DAYS}")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info(f"   Block Size: {BLOCK_SIZE}")
    logger.info(f"   Max Points: {N_POINTS_MAX}")
    logger.info(f"   Data Format: {DATA_FORMAT}")
    if GENERATE_MAPS:
        logger.info(f"   🗺️  Percentile maps: enabled")
        logger.info(f"      Percentiles: {PERCENTILES}")
        logger.info(f"      Days: {MAP_DAYS}")
        logger.info(f"      Output: {MAP_OUTPUT_DIR}")
    else:
        logger.info(f"   🗺️  Percentile maps: disabled")
    logger.info("=" * 80)

    try:
        # ============================================================
        # ۱. ساخت Adapter و دریافت اطلاعات
        # ============================================================
        year_list = list(range(YEAR_START, YEAR_END + 1))
        adapter = create_adapter(
            ZARR_BASE,
            year_list,
            data_format=DATA_FORMAT,
            cache_enabled=True,
            max_points=N_POINTS_MAX,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
        )

        n_stations = adapter.n_points
        coords = adapter.get_coords()
        station_ids = coords.get("stationid", np.arange(n_stations))
        lons = coords["lon"]
        lats = coords["lat"]
        elevs = coords["elev"]

        logger.info(f"   ✅ {n_stations:,} stations loaded")
        logger.info(f"   ✅ Data format: {DATA_FORMAT}")

        # ============================================================
        # ۲. جداول
        # ============================================================
        logger.info("Building tables...")
        doy_table, _ = build_doy_table_from_config()
        tables = build_runtime_tables(ZARR_BASE)
        window_table = tables["window_table"]
        logger.info("   ✅ Tables built")

        # ============================================================
        # ۳. ایجاد/دریافت Zarr (با مدیریت خودکار resize)
        # ============================================================
        root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   📂 Zarr store ready: {OUTPUT_ZARR}")

        # ============================================================
        # ۴. ✅ استفاده از checkpoint_manager با تشخیص خودکار
        # ============================================================
        checkpoint = ensure_checkpoint(auto_detect=True)
        start_block = checkpoint.get("block", 0)
        start_station = checkpoint.get("station", 0)
        logger.info(f"   ⏩ Resuming from block {start_block}, station {start_station}")

        # ============================================================
        # ۵. حلقه پردازش
        # ============================================================
        total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
        logger.info(f"   Total blocks: {total_blocks}")

        for block_idx in range(start_block, total_blocks):
            block_start = block_idx * BLOCK_SIZE
            block_end = min(block_start + BLOCK_SIZE, n_stations)

            last_station = None
            if block_idx == start_block and start_station > block_start:
                last_station = start_station

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
                    var_idx=VAR_INDEX_FOR_FIT,
                    last_checkpoint_station=last_station,
                    adapter=adapter,
                )
                save_checkpoint(block_idx, block_end - 1)
                gc.collect()
            except KeyboardInterrupt:
                logger.warning(f"Interrupted at block {block_idx}")
                save_checkpoint(block_idx, block_start)
                raise
            except Exception as e:
                logger.error(f"Block {block_idx} failed: {e}")
                save_checkpoint(block_idx, block_start)
                raise

        # ============================================================
        # ۶. نهایی‌سازی
        # ============================================================
        logger.info("Finalizing Zarr...")
        ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}"
        ds.attrs["version"] = "3.3"
        ds.attrs["data_format"] = DATA_FORMAT
        ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
        ds.close()

        delete_checkpoint()
        logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")

        # ============================================================
        # ۷. تولید نقشه‌های صدک (اختیاری)
        # ============================================================
        if GENERATE_MAPS:
            generate_percentile_maps(
                zarr_path=OUTPUT_ZARR,
                output_dir=MAP_OUTPUT_DIR,
                percentiles=PERCENTILES,
                days=MAP_DAYS if MAP_DAYS else None
            )

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if dask_client:
            try:
                dask_client.close()
            except:
                pass

if __name__ == "__main__":
    main()