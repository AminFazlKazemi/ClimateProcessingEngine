#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - نقطه ورود اصلی
نسخه نهایی - بهینه‌شده با کش کردن دیتاست‌ها و پردازش موازی
"""

import os
import sys
import time
import json
import glob
import yaml
import gc
import numpy as np
import xarray as xr

# اضافه کردن مسیر فعلی به sys.path
sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# Import ماژول‌های داخلی
# ============================================================
from constants import (
    YEAR_START, YEAR_END, N_YEARS, N_DAYS,
    VARS, N_VARS, VAR_INDEX_FOR_FIT,
    OUTPUT_DIR, OUTPUT_ZARR, CHECKPOINT_FILE, ZARR_BASE,
    BLOCK_SIZE, USE_PARALLEL, CORES,
    VALIDATE_AFTER_LOAD, VALIDATE_BEFORE_WRITE, VALIDATE_EVERY_N_BLOCKS,
    LOG_LEVEL, LOG_FILE,
    FLOAT_DTYPE, INT_DTYPE
)
from calendar_tables import build_doy_table_from_config
from runtime_tables import build_runtime_tables
from zarr_schema import create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.checkpoint import load_checkpoint, save_checkpoint, delete_checkpoint
from monitoring.logger import logger

# ============================================================
# تنظیمات محیطی برای پردازش موازی
# ============================================================
if USE_PARALLEL:
    os.environ["PARALLEL_WORKERS"] = str(CORES)
    logger.info(f"   🚀 Parallel processing enabled with {CORES} workers")
else:
    os.environ["PARALLEL_WORKERS"] = "1"
    logger.info("   🐢 Parallel processing disabled")

# ============================================================
# تنظیمات Dask (اختیاری - برای سرعت بیشتر)
# ============================================================
dask_client = None
try:
    from dask.distributed import Client
    dask_client = Client(processes=False, threads_per_worker=CORES)
    logger.info(f"   🚀 Dask Client started: {dask_client}")
except ImportError:
    logger.info("   ⚠️ Dask not installed. Using native processing.")
except Exception as e:
    logger.warning(f"   ⚠️ Dask Client failed to start: {e}")

# ============================================================
# توابع کمکی
# ============================================================
def get_station_info(zarr_base):
    """دریافت اطلاعات ایستگاه‌ها از فایل‌های Zarr"""
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"No Zarr files found in {zarr_base}")
    
    ds = xr.open_zarr(zarr_files[0])
    n_stations = ds.sizes["point"]
    station_ids = ds["stationid"].values if "stationid" in ds else np.arange(n_stations)
    lons = ds["lon"].values if "lon" in ds else np.full(n_stations, np.nan)
    lats = ds["lat"].values if "lat" in ds else np.full(n_stations, np.nan)
    elevs = ds["elev"].values if "elev" in ds else np.full(n_stations, np.nan)
    ds.close()
    return n_stations, station_ids, lons, lats, elevs

# ============================================================
# تابع اصلی
# ============================================================
def main():
    """اجرای کل فرایند پردازش"""
    
    logger.info("=" * 80)
    logger.info("🚀 CLIMATOLOGY PROCESSING ENGINE v2.1 - FINAL")
    logger.info(f"   Years: {YEAR_START}–{YEAR_END} ({N_YEARS} years)")
    logger.info(f"   Days: {N_DAYS}")
    logger.info(f"   Variables: {VARS}")
    logger.info(f"   Output: {OUTPUT_ZARR}")
    logger.info(f"   Block Size: {BLOCK_SIZE}")
    logger.info(f"   Parallel: {USE_PARALLEL} (cores={CORES})")
    logger.info(f"   Validation: {VALIDATE_AFTER_LOAD}/{VALIDATE_BEFORE_WRITE}")
    logger.info("=" * 80)
    
    try:
        # ============================================================
        # ۱. دریافت اطلاعات ایستگاه‌ها
        # ============================================================
        logger.info("Reading station information...")
        n_stations, station_ids, lons, lats, elevs = get_station_info(ZARR_BASE)
        logger.info(f"   ✅ {n_stations:,} stations found")
        
        # ============================================================
        # ۲. ساخت جداول تقویم
        # ============================================================
        logger.info("Building calendar tables...")
        doy_table, _ = build_doy_table_from_config()
        logger.info(f"   ✅ doy_table shape: {doy_table.shape}")
        
        # ============================================================
        # ۳. ساخت جداول زمان اجرا
        # ============================================================
        logger.info("Building runtime tables...")
        tables = build_runtime_tables(ZARR_BASE)
        file_map = tables["file_map"]
        window_table = tables["window_table"]
        year_list = tables["year_list"]
        logger.info(f"   ✅ {len(file_map)} files in file_map")
        logger.info(f"   ✅ {len(window_table)} windows in window_table")
        
        # ============================================================
        # ۴. تنظیم block_size (بدون Benchmark)
        # ============================================================
        actual_block_size = BLOCK_SIZE
        logger.info(f"ℹ️ Using block size: {actual_block_size} (from config.yaml)")
        
        # ============================================================
        # ۵. ایجاد Zarr Store
        # ============================================================
        logger.info("Creating Zarr store...")
        root = create_zarr_store(OUTPUT_ZARR, n_stations)
        logger.info(f"   ✅ Zarr created: {OUTPUT_ZARR}")
        
        # ============================================================
        # ۶. بارگذاری Checkpoint
        # ============================================================
        checkpoint = load_checkpoint()
        start_block = 0
        start_station = 0
        
        if checkpoint:
            start_block = checkpoint.get("block", 0)
            start_station = checkpoint.get("station", 0)
            logger.info(f"   ⏩ Resuming from block {start_block}, station {start_station}")
        else:
            logger.info("   🆕 Starting from beginning")
        
        # ============================================================
        # ۷. حلقه اصلی پردازش
        # ============================================================
        total_blocks = (n_stations + actual_block_size - 1) // actual_block_size
        logger.info("Starting processing...")
        logger.info(f"   Total blocks: {total_blocks}")
        logger.info(f"   Starting from block {start_block}")
        logger.info(f"   Block size: {actual_block_size}")
        
        total_start_time = time.time()
        
        for block_idx in range(start_block, total_blocks):
            block_start = block_idx * actual_block_size
            block_end = min(block_start + actual_block_size, n_stations)
            
            # تنظیم نقطه شروع برای اولین بلوک
            last_station = None
            if block_idx == start_block and start_station > 0:
                last_station = start_station
            
            try:
                # پردازش بلوک
                result = process_block(
                    block_start=block_start,
                    block_end=block_end,
                    block_idx=block_idx,
                    file_map=file_map,
                    doy_table=doy_table,
                    window_table=window_table,
                    year_list=year_list,
                    root=root,
                    var_idx=VAR_INDEX_FOR_FIT,
                    last_checkpoint_station=last_station,
                )
                # ذخیره Checkpoint پس از موفقیت
                save_checkpoint(block_idx, block_end - 1)
                
                # آزادسازی حافظه بعد از هر بلوک
                gc.collect()
                
            except KeyboardInterrupt:
                logger.warning(f"Interrupted at block {block_idx}. Checkpoint saved.")
                save_checkpoint(block_idx, block_start)
                raise
            except Exception as e:
                logger.error(f"Block {block_idx} failed: {e}")
                save_checkpoint(block_idx, block_start)
                raise
        
        # ============================================================
        # ۸. نهایی‌سازی
        # ============================================================
        logger.info("Finalizing Zarr...")
        ds = xr.open_zarr(OUTPUT_ZARR)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}, Window ±{WINDOW_DAYS} days"
        ds.attrs["variables"] = VARS
        ds.attrs["fit_variable_index"] = VAR_INDEX_FOR_FIT
        ds.attrs["block_size"] = actual_block_size
        ds.to_zarr(OUTPUT_ZARR, mode="w", consolidated=True)
        
        # پاک کردن کش دیتاست‌ها
        try:
            from io_pipeline.read_month_files import clear_ds_cache
            clear_ds_cache()
        except:
            pass
        
        # حذف Checkpoint
        delete_checkpoint()
        
        total_time = time.time() - total_start_time
        logger.info("=" * 80)
        logger.info("✅ PROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"   Total time: {total_time / 60:.1f} minutes ({total_time:.1f} seconds)")
        logger.info(f"   Output: {OUTPUT_ZARR}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Checkpoint saved.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # بستن Dask Client در صورت وجود
        if dask_client:
            try:
                dask_client.close()
                logger.info("   Dask Client closed.")
            except:
                pass

# ============================================================
# اجرای اصلی
# ============================================================
if __name__ == "__main__":
    main()