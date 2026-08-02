#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_limited_fixed.py – پردازش فقط نقاط ۰ تا ۳۸۰۰۰ (امن)
================================================================================
- از Zarr موجود استفاده می‌کند و آن را پاک نمی‌کند.
- فقط بلوک‌های ۰ تا ۱۹ را پردازش می‌کند.
- با قفل برای جلوگیری از تداخل با build_cache_only.
================================================================================
"""

import os
import sys
import time
import gc
import yaml
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    YEAR_START, YEAR_END, VAR_INDEX_FOR_FIT,
    OUTPUT_ZARR, ZARR_BASE, BLOCK_SIZE,
)
from calendar_tables import build_doy_table_from_config
from runtime_tables import build_runtime_tables
from zarr_schema import get_or_create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger
from data_adapter import create_adapter
from lock_utils import acquire_lock, release_lock, is_locked, clean_stale_locks

# ============================================================
# بارگذاری config
# ============================================================
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

MAX_POINTS_TO_PROCESS = 38000
MAX_BLOCK = (MAX_POINTS_TO_PROCESS + BLOCK_SIZE - 1) // BLOCK_SIZE - 1  # = 18

logger.info("=" * 80)
logger.info("🚀 MAIN LIMITED (امن) – فقط نقاط ۰ تا ۳۸۰۰۰")
logger.info(f"   - نقاط ۰ تا {MAX_POINTS_TO_PROCESS - 1}")
logger.info(f"   - بلوک‌های ۰ تا {MAX_BLOCK}")
logger.info("   - Zarr موجود پاک نمی‌شود")
logger.info("=" * 80)

# پاک کردن قفل‌های قدیمی
clean_stale_locks(timeout_seconds=300)

# ============================================================
# آماده‌سازی
# ============================================================
year_list = list(range(YEAR_START, YEAR_END + 1))
adapter = create_adapter(ZARR_BASE, year_list, cache_enabled=True)
n_stations = adapter.n_points
logger.info(f"📍 تعداد کل نقاط در Zarr: {n_stations:,}")

# ============================================================
# بررسی Zarr بدون دستکاری
# ============================================================
if not os.path.exists(OUTPUT_ZARR):
    logger.error(f"❌ Zarr خروجی وجود ندارد: {OUTPUT_ZARR}")
    sys.exit(1)

# ============================================================
# جداول
# ============================================================
doy_table, _ = build_doy_table_from_config()
tables = build_runtime_tables(ZARR_BASE)
window_table = tables["window_table"]

# ============================================================
# باز کردن Zarr (با n_stations واقعی)
# ============================================================
root = get_or_create_zarr_store(OUTPUT_ZARR, n_stations)

# ============================================================
# حلقه‌ی پردازش – فقط تا MAX_BLOCK
# ============================================================
start_block = 0
start_station = 0

from checkpoint_manager import ensure_checkpoint, save_checkpoint
checkpoint = ensure_checkpoint(auto_detect=False)
if checkpoint and checkpoint.get("block", 0) <= MAX_BLOCK:
    start_block = checkpoint["block"]
    start_station = checkpoint["station"]

logger.info(f"📦 شروع از بلوک {start_block} (تا {MAX_BLOCK})")

for block_idx in range(start_block, MAX_BLOCK + 1):
    block_start = block_idx * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_stations, MAX_POINTS_TO_PROCESS)
    
    if block_start >= MAX_POINTS_TO_PROCESS:
        break

    if is_locked(block_idx):
        logger.info(f"🔒 بلوک {block_idx} قفل است – صبر می‌کنیم...")
        while is_locked(block_idx):
            time.sleep(1)
        logger.info(f"🔓 بلوک {block_idx} آزاد شد")

    if not acquire_lock(block_idx, timeout=10):
        logger.warning(f"⏳ نمی‌توان قفل بلوک {block_idx} را گرفت – رد می‌شود")
        continue

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
            last_checkpoint_station=start_station if block_idx == start_block else None,
            adapter=adapter,
        )
        save_checkpoint(block_idx, block_end - 1)
        gc.collect()
    finally:
        release_lock(block_idx)

logger.info("🔚 پردازش محدوده‌ی ۰ تا ۳۸۰۰۰ با موفقیت کامل شد!")
logger.info(f"📁 خروجی: {OUTPUT_ZARR}")