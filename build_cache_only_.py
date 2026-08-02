#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cache_only.py – ساخت کش با پرش از بلوک‌های قفل‌شده
================================================================================
- فقط بلوک‌های ۰ تا ۱۹ (نقاط ۰ تا ۳۷۹۹۹) را پردازش می‌کند.
- اگر یک بلوک قفل باشد (یعنی main.py در حال پردازش آن است)، از آن عبور می‌کند.
- هر بلوک را با قفل جداگانه محافظت می‌کند.
================================================================================
"""

import os
import sys
import time
import gc
import yaml
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    ZARR_BASE, YEAR_START, YEAR_END, BLOCK_SIZE, CHECKPOINT_FILE
)
from runtime_tables import build_runtime_tables
from data_adapter import create_adapter
from monitoring.logger import logger
from lock_utils import acquire_lock, release_lock, is_locked, clean_stale_locks

# ============================================================
# بارگذاری config
# ============================================================
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

LAT_MIN = CONFIG.get("lat_min")
LAT_MAX = CONFIG.get("lat_max")
LON_MIN = CONFIG.get("lon_min")
LON_MAX = CONFIG.get("lon_max")
DATA_FORMAT = CONFIG.get("data_format", "auto")
N_POINTS_MAX = CONFIG.get("n_sample_points", 40000)

# ============================================================
# توابع کمکی
# ============================================================
def cache_exists(adapter, block_start, block_size, year_idx, month, sample_hash="all"):
    possible_block_sizes = [1000, 2000, 5000]
    possible_hashes = ["all", sample_hash]
    
    for bs in possible_block_sizes:
        adjusted_start = (block_start // bs) * bs
        for hs in possible_hashes:
            try:
                key = adapter.cache._get_cache_key(
                    block_start=adjusted_start,
                    block_size=bs,
                    year=year_idx,
                    month=month,
                    var="all_vars",
                    sample_hash=hs
                )
                path = adapter.cache._get_cache_path(key)
                if os.path.exists(path):
                    return True
            except Exception:
                continue
    return False

def get_checkpoint():
    """خواندن checkpoint از فایل"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    data = {}
    with open(CHECKPOINT_FILE, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                data[k] = v
    if "block" in data and "station" in data:
        return {"block": int(data["block"]), "station": int(data["station"])}
    return None

def save_checkpoint(block_idx, station_idx):
    """ذخیره checkpoint"""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(f"block={block_idx}\n")
        f.write(f"station={station_idx}\n")
        f.write(f"timestamp={int(time.time())}\n")

# ============================================================
# تابع اصلی
# ============================================================
def build_cache():
    logger.info("=" * 80)
    logger.info("🚀 BUILD CACHE – با پرش از بلوک‌های قفل‌شده")
    logger.info("   - بلوک‌های ۰ تا ۱۹ (نقاط ۰ تا ۳۷۹۹۹)")
    logger.info("   - اگر بلوکی قفل باشد، از آن عبور می‌کند")
    logger.info("=" * 80)

    # پاک کردن قفل‌های قدیمی (بیش از ۵ دقیقه)
    clean_stale_locks(timeout_seconds=300)

    # ============================================================
    # تنظیمات اولیه
    # ============================================================
    year_list = list(range(YEAR_START, YEAR_END + 1))
    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]

    adapter = create_adapter(
        zarr_base=ZARR_BASE,
        year_list=year_list,
        data_format=DATA_FORMAT,
        cache_enabled=True,
        max_points=N_POINTS_MAX,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        lon_min=LON_MIN,
        lon_max=LON_MAX
    )

    n_stations = adapter.n_points
    sample_hash = adapter._get_sample_hash()

    # محدوده‌ی هدف
    TARGET_END_BLOCK = min(19, (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE - 1)
    
    # شروع از checkpoint (اگر معتبر باشد)
    checkpoint = get_checkpoint()
    start_block = 0
    if checkpoint and checkpoint.get("block", 0) <= TARGET_END_BLOCK:
        start_block = checkpoint["block"]
        logger.info(f"📌 شروع از checkpoint: بلوک {start_block}")

    logger.info(f"📍 نقاط کل: {n_stations:,}")
    logger.info(f"📦 بلوک‌های هدف: ۰ تا {TARGET_END_BLOCK}")

    start_time = time.time()
    total_files_checked = 0
    total_files_missing = 0
    total_files_loaded = 0
    skipped_locked = 0

    # ============================================================
    # حلقه‌ی اصلی
    # ============================================================
    for block_idx in range(start_block, TARGET_END_BLOCK + 1):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        block_size_actual = block_end - block_start

        # ============================================================
        # ۱. بررسی قفل
        # ============================================================
        if is_locked(block_idx):
            logger.info(f"🔒 بلوک {block_idx} قفل است (main.py در حال پردازش) – عبور می‌کنیم")
            skipped_locked += 1
            continue

        # ============================================================
        # ۲. گرفتن قفل
        # ============================================================
        if not acquire_lock(block_idx, timeout=2):
            logger.info(f"⏳ بلوک {block_idx} قفل شد – عبور می‌کنیم")
            skipped_locked += 1
            continue

        try:
            logger.info(f"\n📦 بلوک {block_idx + 1}/{TARGET_END_BLOCK + 1}: "
                        f"ایستگاه‌های {block_start:,}–{block_end:,}")

            months_to_load = []
            for year in year_list:
                for month in range(1, 13):
                    if (year, month) in file_map:
                        months_to_load.append((year, month))

            pbar = tqdm(months_to_load, desc="   بررسی و ساخت کش", unit="file")

            for year, month in months_to_load:
                year_idx = year_list.index(year) if year in year_list else 0
                total_files_checked += 1

                if cache_exists(adapter, block_start, block_size_actual, year_idx, month, sample_hash):
                    pbar.update(1)
                    continue

                total_files_missing += 1
                combined_data = adapter.load_block_all_vars(
                    block_start=block_start,
                    block_size=block_size_actual,
                    year_idx=year_idx,
                    month=month
                )

                if combined_data is not None:
                    total_files_loaded += 1
                else:
                    logger.warning(f"   ⚠️ بارگذاری ناموفق: {year}-{month:02d}")

                del combined_data
                gc.collect()
                pbar.update(1)

            pbar.close()
            save_checkpoint(block_idx, block_end - 1)

        finally:
            # همیشه قفل را آزاد کن
            release_lock(block_idx)

        elapsed = time.time() - start_time
        logger.info(f"   ✅ بلوک {block_idx + 1} کامل شد. "
                    f"زمان: {elapsed:.1f}s | "
                    f"کش‌های جدید: {total_files_loaded}")

    # ============================================================
    # گزارش نهایی
    # ============================================================
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✅ ساخت کش با موفقیت کامل شد!")
    logger.info(f"   کش‌های موجود: {total_files_checked - total_files_missing}")
    logger.info(f"   کش‌های جدید: {total_files_loaded}")
    logger.info(f"   بلوک‌های قفل‌شده (رد شده): {skipped_locked}")
    logger.info(f"   زمان کل: {total_time:.1f} ثانیه")
    logger.info("=" * 80)

if __name__ == "__main__":
    build_cache()