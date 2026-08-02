#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cache_only_reverse.py – ساخت کش از ۳۸۰۰۰ به عقب
================================================================================
- از بلوک ۱۹ (نقطه ۳۸۰۰۰) شروع می‌کند و به سمت بلوک ۰ (نقطه ۰) برمی‌گردد.
- فقط بلوک‌های ۰ تا ۱۹ را پردازش می‌کند.
- از checkpoint_manager برای ادامه‌ی پردازش استفاده می‌کند.
- اگر checkpoint بلوک بزرگتر از ۱۹ را نشان دهد، آن را نادیده می‌گیرد.
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

# ============================================================
# ۱. بارگذاری config.yaml
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

LAT_MIN = CONFIG.get("lat_min")
LAT_MAX = CONFIG.get("lat_max")
LON_MIN = CONFIG.get("lon_min")
LON_MAX = CONFIG.get("lon_max")
DATA_FORMAT = CONFIG.get("data_format", "auto")
N_POINTS_MAX = CONFIG.get("n_sample_points", 40000)

# ============================================================
# ۲. import از constants و checkpoint_manager
# ============================================================
from constants import (
    ZARR_BASE,
    YEAR_START,
    YEAR_END,
    VARS,
    BLOCK_SIZE,
    OUTPUT_ZARR,
    CHECKPOINT_FILE,
)
from runtime_tables import build_runtime_tables
from data_adapter import create_adapter
from monitoring.logger import logger
from checkpoint_manager import ensure_checkpoint, save_checkpoint, delete_checkpoint


def cache_exists(adapter, block_start, block_size, year_idx, month, sample_hash="all"):
    """بررسی وجود فایل کش"""
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


def build_cache_reverse():
    """
    حلقه‌ی اصلی ساخت کش – از بلوک ۱۹ به سمت ۰
    """
    logger.info("=" * 80)
    logger.info("🚀 BUILD CACHE (معکوس) – از ۳۸۰۰۰ به عقب")
    logger.info("   - از بلوک ۱۹ (نقطه ۳۸۰۰۰) شروع می‌کند و به سمت ۰ می‌رود.")
    logger.info("   - فقط بلوک‌های ۰ تا ۱۹ (نقاط ۰ تا ۳۷۹۹۹) پردازش می‌شوند.")
    logger.info("   - بلوک‌های ۲۰ به بعد (نقاط ۴۰۰۰۰+) نادیده گرفته می‌شوند.")
    logger.info("=" * 80)

    # ============================================================
    # بررسی checkpoint – اگر بلوک > ۱۹ است، آن را نادیده بگیر
    # ============================================================
    checkpoint = ensure_checkpoint(auto_detect=False)
    
    # محدوده‌ی هدف: بلوک‌های ۰ تا ۱۹
    TARGET_END_BLOCK = 19
    TARGET_START_BLOCK = 0
    
    # تعیین نقطه‌ی شروع (از انتها به ابتدا)
    if checkpoint and checkpoint.get("block", 0) > TARGET_END_BLOCK:
        logger.warning(f"⚠️ checkpoint بلوک {checkpoint['block']} را نشان می‌دهد که بالاتر از محدوده‌ی هدف (۱۹) است.")
        logger.warning("🔄 نادیده گرفتن checkpoint و شروع از انتها (بلوک ۱۹، ایستگاه آخر بلوک)")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("🗑️ checkpoint.csv پاک شد.")
        # از انتها شروع می‌کنیم
        start_block = TARGET_END_BLOCK
        start_station = (start_block + 1) * BLOCK_SIZE - 1  # آخرین ایستگاه بلوک
    elif checkpoint and checkpoint.get("block", 0) <= TARGET_END_BLOCK:
        # از checkpoint موجود استفاده می‌کنیم، اما از همان بلوک به سمت پایین می‌رویم
        start_block = checkpoint["block"]
        start_station = checkpoint["station"]
        logger.info(f"📌 شروع از checkpoint موجود: بلوک {start_block}, ایستگاه {start_station}")
    else:
        # از انتها شروع می‌کنیم
        start_block = TARGET_END_BLOCK
        start_station = (start_block + 1) * BLOCK_SIZE - 1
        logger.info("🔍 هیچ checkpoint معتبری یافت نشد. شروع از انتها (بلوک ۱۹، ایستگاه آخر)")

    # ============================================================
    # آماده‌سازی سال‌ها و جداول
    # ============================================================
    year_list = list(range(YEAR_START, YEAR_END + 1))
    logger.info(f"📅 سال‌ها: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    logger.info(f"📂 تعداد فایل‌های Zarr پیدا شده: {len(file_map)}")

    # ============================================================
    # ساخت Adapter
    # ============================================================
    logger.info("🔧 ساخت Data Adapter...")
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
    total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE

    # محدوده‌ی هدف: فقط تا ۱۹
    actual_end_block = min(TARGET_END_BLOCK, total_blocks - 1)

    logger.info(f"📍 تعداد کل نقاط: {n_stations:,}")
    logger.info(f"📦 اندازه‌ی بلوک: {BLOCK_SIZE}")
    logger.info(f"📦 بلوک‌های هدف: ۰ تا {actual_end_block} (نقاط ۰ تا {(actual_end_block + 1) * BLOCK_SIZE - 1})")
    logger.info(f"📦 مسیر پردازش: از بلوک {start_block} به سمت ۰")

    sample_hash = adapter._get_sample_hash()

    start_time = time.time()
    total_files_checked = 0
    total_files_missing = 0
    total_files_loaded = 0

    # ============================================================
    # حلقه‌ی اصلی – از start_block به سمت ۰
    # ============================================================
    # محدوده‌ی حلقه: از start_block تا ۰ (با گام -۱)
    for block_idx in range(start_block, -1, -1):
        # اگر از checkpoint شروع کرده‌ایم و در همان بلوک هستیم، از station بعدی ادامه می‌دهیم
        # اما چون کش‌ها مستقل از ایستگاه هستند، کل بلوک را پردازش می‌کنیم
        
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        block_size_actual = block_end - block_start

        logger.info(f"\n📦 بلوک {block_idx + 1}/{actual_end_block + 1} (معکوس): "
                    f"ایستگاه‌های {block_start:,}–{block_end:,} ({block_size_actual} ایستگاه)")

        months_to_load = []
        for year in year_list:
            for month in range(1, 13):
                if (year, month) in file_map:
                    months_to_load.append((year, month))

        pbar = tqdm(months_to_load, desc=f"   بررسی و ساخت کش", unit="file", position=0, leave=True)

        for year, month in months_to_load:
            year_idx = year_list.index(year) if year in year_list else 0

            total_files_checked += 1

            # بررسی وجود کش
            if cache_exists(adapter, block_start, block_size_actual, year_idx, month, sample_hash):
                pbar.update(1)
                continue

            # کش وجود ندارد – از Zarr بخوان و کش جدید بساز
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

        # ذخیره checkpoint بعد از اتمام بلوک (با اندیس بلوک فعلی)
        save_checkpoint(block_idx, block_end - 1)

        elapsed = time.time() - start_time
        logger.info(f"   ✅ بلوک {block_idx + 1} کامل شد. "
                    f"زمان کل: {elapsed:.1f} ثانیه | "
                    f"میانگین: {elapsed / ((start_block - block_idx) + 1):.1f} ثانیه/بلوک")
        logger.info(f"      کش‌های موجود: {total_files_checked - total_files_missing} | "
                    f"کش‌های جدید: {total_files_loaded}")

        # اگر به بلوک ۰ رسیدیم، حلقه تمام می‌شود
        if block_idx == 0:
            break

    # ============================================================
    # گزارش نهایی
    # ============================================================
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✅ ساخت کش (معکوس) با موفقیت کامل شد!")
    logger.info(f"   تعداد کل فایل‌های بررسی‌شده: {total_files_checked:,}")
    logger.info(f"   تعداد فایل‌های بدون کش (نیاز به ساخت): {total_files_missing:,}")
    logger.info(f"   تعداد فایل‌های بارگذاری‌شده (کش‌های جدید): {total_files_loaded:,}")
    logger.info(f"   زمان کل: {total_time:.1f} ثانیه ({total_time / 60:.1f} دقیقه)")
    logger.info("=" * 80)

    # نمایش مسیر کش
    cache_path = os.path.join(os.path.dirname(__file__), "cache")
    if os.path.exists(cache_path):
        import glob
        cache_files = glob.glob(os.path.join(cache_path, "*.pkl.gz"))
        total_size = sum(os.path.getsize(f) for f in cache_files) / (1024**3)
        logger.info(f"📁 تعداد فایل‌های کش: {len(cache_files):,}")
        logger.info(f"💾 حجم کل کش: {total_size:.2f} GB")
        logger.info("=" * 80)

    logger.info("📌 بعد از اتمام ساخت کش، main.py را اجرا کنید تا داده‌ها در Zarr نوشته شوند.")
    logger.info("   (فقط نقاط خالی بازنویسی می‌شوند و بقیه دست نمی‌خورند.)")
    logger.info("=" * 80)


def main():
    try:
        build_cache_reverse()
    except KeyboardInterrupt:
        logger.warning("\n⏹️ کاربر متوقف کرد.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ خطا: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()