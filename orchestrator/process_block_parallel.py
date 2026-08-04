# -*- coding: utf-8 -*-
"""
process_block.py - پردازش یک بلوک از ایستگاه‌ها (نسخه‌ی موازی بهینه)
با استفاده از ThreadPoolExecutor برای پردازش همزمان ایستگاه‌ها،
همراه با نمایش پیشرفت و مدیریت حافظه
"""

import time
import numpy as np
import gc
from tqdm import tqdm
from monitoring.logger import logger
from monitoring.checkpoint import save_checkpoint
from io_pipeline.read_month_files import read_month_files
from io_pipeline.assemble_block import assemble_block
from io_pipeline.validate_block import validate_block
from numerical_engine.merge_results import create_and_merge_results
from result_pipeline.validate_result import validate_result
from result_pipeline.write_block import write_block_safe
from constants import VARS, VAR_INDEX_FOR_FIT, N_DAYS
from zarr_schema import VAR_NAMES, VAR_DTYPES, create_empty_block_result


def process_block(block_start, block_end, block_idx, file_map, doy_table, window_table,
                  year_list, root, var_idx, last_checkpoint_station=0, adapter=None):
    """
    پردازش یک بلوک از ایستگاه‌ها (با موازی‌سازی و مدیریت حافظه).

    پارامترها:
        block_start: شاخص شروع ایستگاه‌ها در بلوک
        block_end: شاخص پایان ایستگاه‌ها (اختصاصی)
        block_idx: شماره بلوک
        file_map: دیکشنری نگاشت (سال, ماه) → مسیر فایل Zarr
        doy_table: جدول روزهای سال (سال‌ها × ۳۶۶)
        window_table: جدول پنجره‌های ۵ روزه برای هر روز سال
        year_list: لیست سال‌ها
        root: گروه Zarr برای نوشتن
        var_idx: شاخص متغیر اصلی برای برازش (معمولاً ۱ = tmean)
        last_checkpoint_station: آخرین ایستگاه پردازش‌شده از checkpoint (برای ادامه)
        adapter: (اختیاری) نمونه‌ای از DataAdapter برای بارگذاری داده

    بازگشت:
        True در صورت موفقیت، None در صورت عدم وجود داده

    استثناها:
        IOError: در صورت خطا در نوشتن
        ValueError: در صورت نامعتبر بودن داده
    """
    block_size = block_end - block_start
    logger.info(f"📦 Block {block_idx}: stations {block_start} - {block_end} ({block_size} stations)")

    times = {"load": 0, "analyze": 0, "write": 0}
    t0 = time.time()

    # ============================================================
    # ۱. خواندن داده‌ها (با یا بدون Adapter)
    # ============================================================
    logger.info("   📂 Loading...")

    if adapter is not None:
        # ============================================================
        # ۱.۱. بارگذاری با استفاده از Data Adapter
        # ============================================================
        logger.info("   📂 Using Data Adapter for loading...")
        data_dict = {}

        # بررسی وجود متد سریع
        if hasattr(adapter, 'load_block_all_vars'):
            logger.info("   ⚡ Using fast multi-variable loading...")
            
            # محاسبه تعداد کل فایل‌هایی که باید خوانده شوند
            total_files = sum(1 for year in year_list for month in range(1, 13) if (year, month) in file_map)
            
            # ایجاد progress bar
            pbar = tqdm(total=total_files, desc="   Loading Zarr files", unit="file", position=0, leave=True)

            for year in year_list:
                for month in range(1, 13):
                    key = (year, month)
                    if key not in file_map:
                        continue
                    combined_data = adapter.load_block_all_vars(
                        block_start=block_start,
                        block_size=block_size,
                        year_idx=year_list.index(year) if year in year_list else 0,
                        month=month
                    )
                    if combined_data is not None:
                        data_dict[key] = combined_data
                    pbar.update(1)
            pbar.close()
        else:
            # روش قدیمی (تک‌متغیره) – برای سازگاری
            logger.info("   🐢 Using fallback single-variable loading...")
            n_vars = len(VARS)
            total_files = sum(1 for year in year_list for month in range(1, 13) if (year, month) in file_map)
            pbar = tqdm(total=total_files, desc="   Loading Zarr files (single-var)", unit="file", position=0, leave=True)

            for year in year_list:
                for month in range(1, 13):
                    key = (year, month)
                    if key not in file_map:
                        continue

                    combined_data = None
                    for v in range(n_vars):
                        var_data = adapter.load_block(
                            block_start=block_start,
                            block_size=block_size,
                            year_idx=year_list.index(year) if year in year_list else 0,
                            month=month,
                            var_idx=v
                        )
                        if var_data is not None:
                            # اطمینان از شکل صحیح
                            if var_data.ndim == 2 and var_data.shape[0] == block_size:
                                var_data = var_data.T
                            elif var_data.ndim == 1:
                                var_data = var_data.reshape(-1, 1)

                            if combined_data is None:
                                days = var_data.shape[0]
                                combined_data = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
                            combined_data[:, :, v] = var_data

                    if combined_data is not None:
                        data_dict[key] = combined_data
                    pbar.update(1)
            pbar.close()

        if not data_dict:
            logger.warning(f"   ⚠️ No data loaded via adapter for block {block_idx}")
            return None

    else:
        # ============================================================
        # ۱.۲. بارگذاری با روش قدیمی (بدون Adapter)
        # ============================================================
        data_dict = read_month_files(block_start, block_size, file_map, year_list)

        if not data_dict:
            logger.warning(f"   ⚠️ No data loaded for block {block_idx}")
            return None

    times["load"] = time.time() - t0

    # ============================================================
    # ۲. مونتاژ داده‌ها
    # ============================================================
    block_data = assemble_block(data_dict, doy_table, block_size, year_list, var_idx)

    # اطمینان از اینکه block_data یک numpy array است
    if hasattr(block_data, 'values'):
        block_data = block_data.values
    elif not isinstance(block_data, np.ndarray):
        block_data = np.array(block_data)

    # ============================================================
    # ۳. اعتبارسنجی بلوک
    # ============================================================
    if block_data.size > 0:
        validate_block(block_data, block_start, block_size, f"Block {block_idx}")

    # ============================================================
    # ۴. تحلیل آماری هر ایستگاه (به صورت موازی با نمایش پیشرفت)
    # ============================================================
    logger.info("   ⚙️ Analyzing (parallel with progress)...")

    t_analyze_start = time.time()

    # تعداد هسته‌های پردازش – محدود به ۴ برای جلوگیری از مصرف بیش از حد حافظه
    import os
    max_workers = min(os.cpu_count() or 4, 4)  # حداکثر ۴ کارگر همزمان
    
    # ============================================================
    # ✅ ارسال year_list به تابع موازی
    # ============================================================
    block_result = create_and_merge_results(
        block_data=block_data,
        year_list=year_list,
        window_table=window_table,
        var_idx=var_idx,
        use_parallel=True,
        n_workers=max_workers
    )

    times["analyze"] = time.time() - t_analyze_start

    # ============================================================
    # ۵. اعتبارسنجی نتایج
    # ============================================================
    if block_result:
        validate_result(block_result, block_start, block_size)

    # ============================================================
    # ۶. نوشتن در Zarr (همزمان – بدون Async)
    # ============================================================
    t0 = time.time()
    try:
        # نوشتن همزمان با async_mode=False
        write_block_safe(root, block_result, block_start, block_end, validate=False, async_mode=False)
        times["write"] = time.time() - t0
    except Exception as e:
        logger.error(f"   ❌ Write failed: {e}")
        save_checkpoint(block_idx, block_start)
        raise IOError(f"Write failed: {e}")

    # ذخیره‌ی چک‌پوینت (بعد از اطمینان از نوشته شدن)
    save_checkpoint(block_idx, block_end - 1)

    # ============================================================
    # ۷. گزارش زمان
    # ============================================================
    total_time = sum(times.values())
    logger.info(f"   ✅ Block {block_idx} completed in {total_time:.1f}s")
    logger.info(f"       Load: {times['load']:.1f}s | Analyze: {times['analyze']:.1f}s | Write: {times['write']:.1f}s")
    if total_time > 0:
        logger.info(f"       Stations/sec: {block_size / total_time:.1f}")

    # آزادسازی حافظه
    del data_dict, block_data, block_result
    gc.collect()

    return True