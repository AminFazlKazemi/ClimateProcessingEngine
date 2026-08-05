#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_block0_status.py - بررسی کامل بودن بلوک ۰ و وضعیت کلی متغیرها
================================================================================
- بررسی می‌کند که آیا بلوک ۰ (ایستگاه‌های ۰ تا ۱۹۹۹) برای هر سه متغیر کامل است.
- نمایش تعداد نقاط معتبر در روز اول برای هر متغیر در بلوک ۰.
- نمایش وضعیت کلی پیشرفت (تعداد بلوک‌های کامل برای هر متغیر).
================================================================================
"""

import os
import numpy as np
import xarray as xr

# ============================================================================
# تنظیمات
# ============================================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000
VARS = ['tmin', 'tmean', 'tmax']
DAY_INDEX = 0  # روز اول

print("=" * 80)
print("🔍 بررسی وضعیت بلوک ۰ و پیشرفت کلی")
print(f"   فایل: {ZARR_PATH}")
print(f"   اندازه بلوک: {BLOCK_SIZE}")
print("=" * 80)

# ============================================================================
# ۱. باز کردن Zarr
# ============================================================================
if not os.path.exists(ZARR_PATH):
    print(f"❌ فایل Zarr وجود ندارد: {ZARR_PATH}")
    exit(1)

ds = xr.open_zarr(ZARR_PATH, consolidated=False)
n_stations = ds.sizes['point']
n_days = ds.sizes['day']
print(f"\n📊 ابعاد: {n_days} روز × {n_stations:,} ایستگاه")

# ============================================================================
# ۲. بررسی بلوک ۰
# ============================================================================
block0_start = 0
block0_end = min(BLOCK_SIZE, n_stations)
print(f"\n📦 بلوک ۰: ایستگاه‌های {block0_start:,} تا {block0_end-1:,}")

block0_complete = True
for var in VARS:
    mean_var = f'{var}_mean'
    if mean_var not in ds:
        print(f"   ⚠️ متغیر {mean_var} در Zarr وجود ندارد!")
        block0_complete = False
        continue
    
    data = ds[mean_var].isel(day=DAY_INDEX).values
    block_data = data[block0_start:block0_end]
    valid_count = np.sum(~np.isnan(block_data))
    total_count = len(block_data)
    is_complete = (valid_count == total_count)
    
    status = "✅ کامل" if is_complete else f"⚠️ {valid_count:,}/{total_count:,}"
    print(f"   {var.upper()}: {status}")
    
    if not is_complete:
        block0_complete = False

if block0_complete:
    print("\n🎉 بلوک ۰ برای همه متغیرها کامل است!")
else:
    print("\n⚠️ بلوک ۰ هنوز کامل نشده است.")

# ============================================================================
# ۳. وضعیت کلی پیشرفت (تعداد بلوک‌های کامل برای هر متغیر)
# ============================================================================
print("\n" + "=" * 80)
print("📊 وضعیت کلی پیشرفت (بر اساس روز اول):")

total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE

for var in VARS:
    mean_var = f'{var}_mean'
    if mean_var not in ds:
        print(f"   ⚠️ {var.upper()}: متغیر وجود ندارد")
        continue
    
    data = ds[mean_var].isel(day=DAY_INDEX).values
    complete_blocks = 0
    first_incomplete = None
    
    for block_idx in range(total_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n_stations)
        block_data = data[start:end]
        valid_count = np.sum(~np.isnan(block_data))
        total_count = len(block_data)
        if valid_count == total_count:
            complete_blocks += 1
        else:
            if first_incomplete is None:
                first_incomplete = block_idx
    
    print(f"   {var.upper()}: {complete_blocks} از {total_blocks} بلوک کامل")
    if first_incomplete is not None:
        print(f"      اولین بلوک ناقص: {first_incomplete}")

# ============================================================================
# ۴. جمع‌بندی
# ============================================================================
print("\n" + "=" * 80)
if block0_complete:
    print("✅ بلوک ۰ کامل است. برنامه به‌درستی کار کرده است.")
else:
    print("⚠️ بلوک ۰ کامل نیست. ممکن است نیاز به بررسی داشته باشد.")
print("=" * 80)

ds.close()