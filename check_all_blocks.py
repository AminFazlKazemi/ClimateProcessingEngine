#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_all_blocks.py – بررسی بلوک به بلوک Zarr نهایی
================================================================================
- فایل Zarr خروجی را باز می‌کند
- هر بلوک را به ترتیب (از ۰ تا آخر) بررسی می‌کند
- برای هر بلوک، وضعیت هر سه متغیر (tmin, tmean, tmax) را گزارش می‌دهد
- تعداد ایستگاه‌های خالی برای هر متغیر را نشان می‌دهد
- در پایان خلاصه بلوک‌های ناقص را نمایش می‌دهد
================================================================================
"""

import os
import numpy as np
import xarray as xr

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000
VARS = ['tmin', 'tmean', 'tmax']

print("=" * 80)
print(f"🔍 بررسی بلوک به بلوک Zarr خروجی")
print(f"   فایل: {ZARR_PATH}")
print(f"   اندازه بلوک: {BLOCK_SIZE}")
print("=" * 80)

if not os.path.exists(ZARR_PATH):
    print(f"❌ فایل Zarr وجود ندارد: {ZARR_PATH}")
    exit(1)

ds = xr.open_zarr(ZARR_PATH, consolidated=False)
n_stations = ds.sizes['point']
n_days = ds.sizes['day']

total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
print(f"📊 ابعاد: {n_days} روز × {n_stations:,} ایستگاه")
print(f"📦 تعداد کل بلوک‌ها: {total_blocks}")
print("=" * 80)

# ============================================================
# بررسی هر بلوک
# ============================================================
incomplete_blocks = {}

for block_idx in range(total_blocks):
    start = block_idx * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n_stations)
    block_size = end - start
    
    block_status = {}
    all_complete = True
    
    for var in VARS:
        mean_var = f'{var}_mean'
        if mean_var not in ds:
            block_status[var] = {'missing_count': block_size, 'missing_indices': list(range(block_size))}
            all_complete = False
            continue
        
        # خواندن همه روزها برای این بلوک
        data = ds[mean_var].isel(day=slice(None)).values[:, start:end]
        is_missing = np.isnan(data).any(axis=0)
        missing_indices = np.where(is_missing)[0].tolist()
        missing_count = len(missing_indices)
        
        block_status[var] = {
            'missing_count': missing_count,
            'missing_indices': missing_indices,
            'is_complete': (missing_count == 0)
        }
        
        if missing_count > 0:
            all_complete = False
    
    incomplete_blocks[block_idx] = {
        'status': block_status,
        'is_complete': all_complete
    }

# ============================================================
# گزارش بلوک‌های ناقص
# ============================================================
print("\n📋 بلوک‌های ناقص (به ترتیب):")
print("-" * 80)

found_incomplete = False
for block_idx, info in incomplete_blocks.items():
    if info['is_complete']:
        continue
    
    found_incomplete = True
    start = block_idx * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, n_stations)
    
    print(f"\n📦 بلوک {block_idx}: ایستگاه‌های {start:,} تا {end-1:,}")
    
    for var in VARS:
        status = info['status'][var]
        missing_count = status['missing_count']
        if missing_count == 0:
            print(f"   {var.upper()}: ✅ کامل")
        else:
            print(f"   {var.upper()}: ⚠️ {missing_count} ایستگاه خالی")
            # نمایش ۵ ایستگاه خالی اول
            if len(status['missing_indices']) <= 10:
                print(f"      ایستگاه‌های خالی: {status['missing_indices']}")
            else:
                print(f"      ایستگاه‌های خالی (۱۰ تای اول): {status['missing_indices'][:10]} ...")

if not found_incomplete:
    print("\n🎉 همه بلوک‌ها برای همه متغیرها کامل هستند!")
else:
    # ============================================================
    # خلاصه نهایی
    # ============================================================
    print("\n" + "=" * 80)
    print("📊 خلاصه بلوک‌های ناقص:")
    print("-" * 80)
    
    for block_idx, info in incomplete_blocks.items():
        if info['is_complete']:
            continue
        
        start = block_idx * BLOCK_SIZE
        missing_vars = []
        for var in VARS:
            if info['status'][var]['missing_count'] > 0:
                missing_vars.append(f"{var}: {info['status'][var]['missing_count']}")
        
        print(f"   بلوک {block_idx}: " + ", ".join(missing_vars))
    
    print("\n" + "=" * 80)
    print("📌 اگر می‌خواهید این بلوک‌ها را پردازش کنید:")
    print("   python main.py --no-checkpoint")
    print("   (یا main.py را طوری تنظیم کنید که از ابتدا اسکن کند)")
    print("=" * 80)

ds.close()