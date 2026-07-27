#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_pipeline/read_month_files.py
============================================
خواندن فایل‌های Zarr با کش کردن دیتاست‌های کامل.
هر فایل فقط یک بار باز می‌شود و برای هر بلوک، ایستگاه‌های مورد نظر
با isel استخراج می‌شوند.
"""

import os
import xarray as xr
import numpy as np
from constants import VARS
from monitoring.logger import logger

# ============================================================
# کش سراسری برای دیتاست‌های کامل (فقط متادیتا و مختصات)
# ============================================================
_DS_CACHE = {}

def read_month_files(block_start, block_size, file_map, year_list):
    """
    خواندن داده‌های Zarr برای یک بلوک با کش کردن دیتاست‌ها.
    """
    data_dict = {}
    
    if not file_map:
        logger.warning("file_map خالی است!")
        return data_dict
    
    total_files = len(file_map)
    processed = 0
    cached_count = 0
    
    logger.info(f"   📂 شروع بارگذاری بلوک {block_start:,}-{block_start+block_size:,} از {total_files} فایل...")
    
    for (year, month), file_path in file_map.items():
        try:
            # ============================================================
            # اگر دیتاست در کش نیست، آن را باز کن و در کش ذخیره کن
            # ============================================================
            if file_path not in _DS_CACHE:
                ds = xr.open_zarr(file_path)
                if "point" not in ds.dims:
                    logger.warning(f"   ⚠️ بعد 'point' در {file_path} وجود ندارد!")
                    ds.close()
                    continue
                _DS_CACHE[file_path] = ds
                cached_count += 1
            else:
                ds = _DS_CACHE[file_path]
            
            # ============================================================
            # استخراج ایستگاه‌های مورد نظر (فقط ۵۰۰۰ ایستگاه)
            # ============================================================
            n_points = ds.sizes["point"]
            end_idx = min(block_start + block_size, n_points)
            
            # اگر block_start خارج از محدوده است، ادامه نده
            if block_start >= n_points:
                continue
            
            ds_block = ds.isel(point=slice(block_start, end_idx))
            
            # ============================================================
            # استخراج داده‌های عددی برای هر متغیر
            # ============================================================
            for var_idx, var_name in enumerate(VARS):
                if var_name in ds_block.data_vars:
                    data = ds_block[var_name].values
                    if data.ndim == 3:
                        data = data.squeeze()
                    data_dict[(year, month, var_idx)] = data
            
            processed += 1
            if processed % 100 == 0:
                logger.info(f"   📂 خوانده شد: {processed}/{total_files} فایل (کش: {cached_count})")
                
        except Exception as e:
            logger.error(f"   ❌ خطا در خواندن {file_path}: {e}")
            continue
    
    logger.info(f"   ✅ خواندن {processed}/{total_files} فایل برای بلوک {block_start} با موفقیت انجام شد.")
    logger.info(f"   💾 کش دیتاست‌ها: {len(_DS_CACHE)} فایل")
    return data_dict

def clear_ds_cache():
    """پاک کردن کش دیتاست‌ها (برای آزادسازی حافظه)"""
    global _DS_CACHE
    for ds in _DS_CACHE.values():
        try:
            ds.close()
        except:
            pass
    _DS_CACHE.clear()
    logger.info("   🧹 کش دیتاست‌ها پاک شد")