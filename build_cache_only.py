#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_cache_only.py – ساخت کش برای بلوک‌های بدون کش (با ادامه از checkpoint)
================================================================================
- از checkpoint_manager با تشخیص خودکار برای ادامه استفاده می‌کند.
- فقط وجود فایل کش را بررسی می‌کند (بدون خواندن داده).
- اگر کش وجود داشت، هیچ کاری نمی‌کند و به سراغ بعدی می‌رود.
- اگر کش وجود نداشت، از Zarr می‌خواند و کش جدید با block_size فعلی (۲۰۰۰) می‌سازد.
- فقط بلوک‌های پردازش‌نشده (بر اساس checkpoint) پردازش می‌شوند.
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
)
from runtime_tables import build_runtime_tables
from data_adapter import create_adapter
from monitoring.logger import logger

# ============================================================
# ✅ استفاده از checkpoint_manager با تشخیص خودکار
# ============================================================
from checkpoint_manager import ensure_checkpoint, save_checkpoint, auto_detect_last_valid_point


def cache_exists(adapter, block_start, block_size, year_idx, month, sample_hash="all"):
    """
    بررسی وجود فایل کش برای ترکیب‌های مختلف block_size و sample_hash.
    
    پارامترها:
        adapter: نمونه‌ی DataAdapter
        block_start: اندیس شروع بلوک
        block_size: اندازه‌ی بلوک فعلی (برای تنظیم offset)
        year_idx: اندیس سال
        month: ماه
        sample_hash: هش نمونه‌برداری فعلی
    
    بازگشت:
        True اگر حداقل یکی از کش‌ها وجود داشته باشد، در غیر این صورت False
    """
    # لیست block_sizeهای احتمالی برای جستجو
    possible_block_sizes = [1000, 2000, 5000]
    # sample_hashهای احتمالی
    possible_hashes = ["all", sample_hash]
    
    for bs in possible_block_sizes:
        # محاسبه‌ی block_start متناسب با block_size جدید
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
                    # اگر کشی پیدا شد، True برگردان
                    return True
            except Exception:
                continue
    return False


def build_cache():
    """
    حلقه‌ی اصلی ساخت کش با نادیده گرفتن کامل کش‌های موجود.
    """
    logger.info("=" * 80)
    logger.info("🚀 BUILD CACHE – ساخت کش برای بلوک‌های بدون کش (با نادیده گرفتن کش‌های موجود)")
    logger.info("   - از checkpoint_manager با تشخیص خودکار برای ادامه استفاده می‌کند.")
    logger.info("   - کش را با block_sizeهای ۱۰۰۰، ۲۰۰۰، ۵۰۰۰ و sample_hashهای مختلف بررسی می‌کند.")
    logger.info("   - اگر هر کدام وجود داشت، بلوک را رد می‌کند (هیچ داده‌ای بارگذاری نمی‌شود).")
    logger.info("   - اگر هیچ‌کدام وجود نداشت، از Zarr می‌خواند و کش جدید با block_size فعلی می‌سازد.")
    logger.info("=" * 80)

    # ============================================================
    # ✅ تشخیص خودکار نقطه‌ی شروع (مشابه main.py)
    # ============================================================
    # ابتدا سعی می‌کنیم از فایل checkpoint بخوانیم
    checkpoint = ensure_checkpoint(auto_detect=False)  # فقط فایل را می‌خواند، تشخیص خودکار نمی‌کند
    if checkpoint and checkpoint.get("block", 0) > 0:
        start_block = checkpoint["block"]
        start_station = checkpoint["station"]
        logger.info(f"📌 شروع از checkpoint موجود: بلوک {start_block}, ایستگاه {start_station}")
    else:
        # اگر checkpoint وجود ندارد یا صفر است، تشخیص خودکار از Zarr
        logger.info("🔍 هیچ checkpoint معتبری یافت نشد. تشخیص خودکار از Zarr...")
        detection = auto_detect_last_valid_point(zarr_path=OUTPUT_ZARR)
        if detection["n_valid"] > 0:
            start_block = detection["checkpoint"]["block"]
            start_station = detection["checkpoint"]["station"]
            logger.info(f"📊 تشخیص خودکار: بلوک {start_block}, ایستگاه {start_station} (بر اساس {detection['n_valid']} نقطه‌ی معتبر)")
            # checkpoint را به‌روز می‌کنیم
            save_checkpoint(start_block, start_station)
        else:
            start_block = 0
            start_station = 0
            logger.info("🆕 هیچ نقطه‌ی معتبری در Zarr یافت نشد. شروع از ابتدا (بلوک ۰، ایستگاه ۰)")

    logger.info(f"⏩ شروع از بلوک {start_block}, ایستگاه {start_station}")

    # ============================================================
    # آماده‌سازی سال‌ها و جداول
    # ============================================================
    year_list = list(range(YEAR_START, YEAR_END + 1))
    logger.info(f"📅 سال‌ها: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    logger.info(f"📂 تعداد فایل‌های Zarr پیدا شده: {len(file_map)}")

    # ============================================================
    # ساخت Adapter (دقیقاً مانند main.py)
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

    logger.info(f"📍 تعداد کل نقاط: {n_stations:,}")
    logger.info(f"📦 اندازه‌ی بلوک: {BLOCK_SIZE}")
    logger.info(f"📦 تعداد کل بلوک‌ها: {total_blocks}")

    sample_hash = adapter._get_sample_hash()

    start_time = time.time()
    total_files_checked = 0
    total_files_missing = 0
    total_files_loaded = 0

    # ============================================================
    # حلقه‌ی اصلی – از start_block به بعد
    # ============================================================
    for block_idx in range(start_block, total_blocks):
        block_start = block_idx * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_stations)
        block_size_actual = block_end - block_start

        # اگر در بلوک start_block هستیم و start_station از block_start بزرگ‌تر است،
        # یعنی بخشی از این بلوک قبلاً پردازش شده، اما چون کش‌ها مستقل از ایستگاه هستند،
        # ما کش را برای کل بلوک می‌سازیم (به ازای همه ایستگاه‌های بلوک).
        # بنابراین نیازی به skipp کردن ایستگاه‌ها نیست.

        logger.info(f"\n📦 بلوک {block_idx + 1}/{total_blocks}: "
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

            # ============================================================
            # ۱. بررسی وجود کش با چندین block_size و sample_hash
            # ============================================================
            if cache_exists(adapter, block_start, block_size_actual, year_idx, month, sample_hash):
                # کش موجود است – هیچ کاری نمی‌کنیم و به بعدی می‌رویم
                pbar.update(1)
                continue

            # ============================================================
            # ۲. کش وجود ندارد – از Zarr بخوان و کش جدید بساز
            # ============================================================
            total_files_missing += 1
            
            # بارگذاری از Zarr و ذخیره‌ی خودکار کش (توسط load_block_all_vars)
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

            # آزادسازی حافظه
            del combined_data
            gc.collect()
            pbar.update(1)

        pbar.close()

        # ذخیره checkpoint بعد از اتمام بلوک
        save_checkpoint(block_idx, block_end - 1)

        elapsed = time.time() - start_time
        logger.info(f"   ✅ بلوک {block_idx + 1} کامل شد. "
                    f"زمان کل: {elapsed:.1f} ثانیه | "
                    f"میانگین: {elapsed / (block_idx + 1):.1f} ثانیه/بلوک")
        logger.info(f"      کش‌های موجود: {total_files_checked - total_files_missing} | "
                    f"کش‌های جدید: {total_files_loaded}")

    # ============================================================
    # گزارش نهایی
    # ============================================================
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✅ ساخت کش با موفقیت کامل شد!")
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

    # نمایش مسیر checkpoint
    from constants import CHECKPOINT_FILE
    logger.info(f"📌 آخرین checkpoint: {CHECKPOINT_FILE}")
    logger.info("=" * 80)


def main():
    try:
        build_cache()
    except KeyboardInterrupt:
        logger.warning("\n⏹️ کاربر متوقف کرد.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ خطا: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()