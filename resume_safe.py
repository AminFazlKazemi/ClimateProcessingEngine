#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resume_safe.py - نسخه با مسیر صحیح
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import zarr

# ============================================================================
# اصلاح مسیر: اضافه کردن پوشه climatology_engine
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(BASE_DIR, "climatology_engine")
if os.path.exists(ENGINE_DIR):
    sys.path.insert(0, ENGINE_DIR)
else:
    # اگر در خود climatology_engine هستیم
    sys.path.insert(0, BASE_DIR)

# ============================================================================
# حالا importها کار میکنند
# ============================================================================
from constants import (
    FIT_VARS, VARS,
    YEAR_START, YEAR_END, N_YEARS, N_DAYS,
    ZARR_BASE, OUTPUT_DIR, OUTPUT_ZARR, CHECKPOINT_FILE,
    BLOCK_SIZE, VAR_INDEX_FOR_FIT,
    MAX_VALUES_PER_FIT, MIN_VALID_VALUES,
)
from zarr_schema import N_OUTPUTS
from calendar_tables import build_calendar_tables
from runtime_tables import build_runtime_tables
from zarr_schema import create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger

# ============================================================================
# ۱. توابع خواندن/نوشتن Checkpoint (مقاوم)
# ============================================================================

def load_checkpoint_safe():
    """خواندن checkpoint با فرمت key=value (بدون JSON)"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    data = {}
    with open(CHECKPOINT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                data[key] = value
    if 'block' in data and 'station' in data:
        return {
            'block': int(data['block']),
            'station': int(data['station']),
            'timestamp': int(data.get('timestamp', 0)),
            'version': int(data.get('version', 0)),
        }
    return None

def save_checkpoint_safe(block_idx, station_idx):
    """ذخیره checkpoint با flush فوری (مقاوم در برابر قطع برق)"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(f"block={block_idx}\n")
        f.write(f"station={station_idx}\n")
        f.write(f"timestamp={int(time.time())}\n")
        f.write(f"version=1\n")
        f.flush()
        os.fsync(f.fileno())  # اطمینان از نوشته شدن روی دیسک

def delete_checkpoint_safe():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

# ============================================================================
# ۲. نوشتن Atomic برای Zarr (مقاوم در برابر قطع برق)
# ============================================================================

def write_block_atomic(root, block_result, block_start, block_end, temp_suffix='.tmp'):
    from zarr_schema import VAR_NAMES
    for name in VAR_NAMES:
        arr = block_result[name]
        # نوشتن مستقیم (Zarr v3 از rename پشتیبانی نمیکند، اما داده را overwrite میکند)
        root[name][:, block_start:block_end] = arr
        if hasattr(root.store, 'flush'):
            root.store.flush()

# ============================================================================
# ۳. تابع اصلی
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("🚀 RESUME SAFE - ادامه پردازش با ذخیره‌سازی مقاوم")
    logger.info("=" * 80)
    
    checkpoint = load_checkpoint_safe()
    if checkpoint:
        start_block = checkpoint['block']
        start_station = checkpoint['station']
        logger.info(f"⏩ ادامه از بلوک {start_block}, ایستگاه {start_station}")
    else:
        start_block = 0
        start_station = 0
        logger.info("🆕 شروع از ابتدا")
    
    logger.info("📂 بارگذاری جداول...")
    calendar_tables = build_calendar_tables()
    doy_table = calendar_tables["doy_table"]
    year_list = calendar_tables["year_list"]
    
    runtime_tables = build_runtime_tables(ZARR_BASE)
    file_map = runtime_tables["file_map"]
    window_table = runtime_tables["window_table"]
    
    logger.info("📊 دریافت اطلاعات ایستگاه‌ها...")
    import glob
    zarr_files = glob.glob(os.path.join(ZARR_BASE, "*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"هیچ فایل Zarr در {ZARR_BASE} یافت نشد")
    ds = xr.open_zarr(zarr_files[0], consolidated=False)
    n_stations = ds.sizes["point"]
    station_ids = ds["stationid"].values
    lons = ds["lon"].values
    lats = ds["lat"].values
    elevs = ds["elev"].values
    ds.close()
    logger.info(f"   ✅ {n_stations:,} ایستگاه")
    
    var_indices = [VARS.index(v) for v in FIT_VARS]
    logger.info(f"   متغیرهای برازش: {FIT_VARS} (indices: {var_indices})")
    
    if os.path.exists(OUTPUT_ZARR) and checkpoint:
        logger.info(f"📂 باز کردن Zarr موجود: {OUTPUT_ZARR}")
        root = zarr.open(OUTPUT_ZARR, mode='a')
    else:
        logger.info(f"🆕 ایجاد Zarr جدید: {OUTPUT_ZARR}")
        root = create_zarr_store(OUTPUT_ZARR, n_stations)
    
    total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
    logger.info(f"📦 تعداد کل بلوک‌ها: {total_blocks}")
    
    for block_idx in range(start_block, total_blocks):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        
        if block_idx == start_block and start_station > block_start:
            last_station = start_station
        else:
            last_station = None
        
        try:
            logger.info(f"\n📦 بلوک {block_idx}/{total_blocks-1}: ایستگاه‌های {block_start:,} - {block_end:,}")
            
            result = process_block(
                block_start=block_start,
                block_end=block_end,
                block_idx=block_idx,
                file_map=file_map,
                doy_table=doy_table,
                window_table=window_table,
                year_list=year_list,
                root=root,
                var_indices=var_indices,
                last_checkpoint_station=last_station,
            )
            
            save_checkpoint_safe(block_idx, block_end - 1)
            logger.info(f"   ✅ checkpoint ذخیره شد (بلوک {block_idx}, ایستگاه {block_end-1})")
            
        except Exception as e:
            logger.error(f"❌ خطا در بلوک {block_idx}: {e}")
            save_checkpoint_safe(block_idx, block_start)
            raise
    
    logger.info("\n🔚 نهایی‌سازی Zarr...")
    root.store.close()
    
    ds = xr.open_zarr(OUTPUT_ZARR, consolidated=False)
    ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
    ds.attrs["source"] = f"Years {YEAR_START}-{YEAR_END}, Window ±2 days"
    ds.attrs["architecture"] = "Station-wise, Block-oriented"
    ds.attrs["version"] = "2.1"
    ds.attrs["n_outputs"] = N_OUTPUTS
    ds.attrs["min_valid_values"] = MIN_VALID_VALUES
    ds.to_zarr(OUTPUT_ZARR, mode="a", consolidated=False)
    ds.close()
    
    try:
        zarr.consolidate_metadata(OUTPUT_ZARR)
    except:
        pass
    
    delete_checkpoint_safe()
    
    logger.info("=" * 80)
    logger.info("✅ پردازش با موفقیت کامل شد!")
    logger.info(f"📁 خروجی: {OUTPUT_ZARR}")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⏹️ کاربر متوقف کرد. checkpoint ذخیره شد.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره: {e}")
        raise