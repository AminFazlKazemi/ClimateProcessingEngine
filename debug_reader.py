#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
عیب‌یابی خواندن داده‌های NetCDF
بررسی ساختار فایل و استخراج داده‌های ۱۰ ایستگاه اول
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import sys
import glob

# ============================================================
# تنظیمات
# ============================================================
BASE_DIR = Path(r"K:/gozareshha/dr vazife/140504 - qc temp/")

# ============================================================
# ۱. پیدا کردن خودکار یک فایل NetCDF نمونه
# ============================================================
print("=" * 60)
print("📂 جستجوی فایل‌های NetCDF در:", BASE_DIR)
print("=" * 60)

nc_files = list(BASE_DIR.rglob("*.nc"))

if not nc_files:
    print("❌ هیچ فایل .nc در دایرکتوری پایه پیدا نشد!")
    print("   لطفاً مسیر دایرکتوری داده‌ها را به‌صورت دستی وارد کنید.")
    user_path = input("📂 مسیر کامل دایرکتوری داده‌ها: ").strip()
    if user_path:
        BASE_DIR = Path(user_path)
        nc_files = list(BASE_DIR.rglob("*.nc"))
    if not nc_files:
        print("❌ باز هم فایلی پیدا نشد. برنامه خاتمه می‌یابد.")
        sys.exit(1)

SAMPLE_FILE = nc_files[0]
print(f"✅ فایل نمونه انتخاب شد: {SAMPLE_FILE}")

# ============================================================
# ۲. بررسی فایل نمونه
# ============================================================
ds = nc.Dataset(SAMPLE_FILE)

print("\n📌 متغیرها:", list(ds.variables.keys()))
print("📌 ابعاد:", ds.dimensions)

# ============================================================
# ۳. شناسه‌های ایستگاه‌ها
# ============================================================
station_var = None
for var_name in ['station', 'station_id', 'id', 'point', 'index']:
    if var_name in ds.variables:
        station_var = var_name
        break

if station_var:
    station_ids = ds.variables[station_var][:]
    print(f"\n✅ متغیر ایستگاه: '{station_var}'")
    print(f"   تعداد ایستگاه‌ها: {len(station_ids)}")
    print(f"   شناسه‌های ۲۰ تای اول: {station_ids[:20]}")
    print(f"   نوع داده: {station_ids.dtype}")
else:
    print("\n⚠️ هیچ متغیر ایستگاه مشخصی یافت نشد.")
    print("   بررسی ابعاد برای یافتن بعد ایستگاه:")
    for dim_name, dim_size in ds.dimensions.items():
        if 'station' in dim_name.lower() or 'point' in dim_name.lower():
            print(f"   بعد '{dim_name}': {dim_size}")

# ============================================================
# ۴. بررسی داده‌های tmin (نمونه)
# ============================================================
var_name = 'tmin'
if var_name in ds.variables:
    data = ds.variables[var_name][:]
    print(f"\n📊 داده‌های {var_name}:")
    print(f"   شکل (shape): {data.shape}")
    print(f"   نوع داده: {data.dtype}")
    
    if data.ndim == 2:
        print("   (زمان, ایستگاه) - ۲ بعدی")
        print(f"   تعداد زمان‌ها: {data.shape[0]}")
        print(f"   تعداد ایستگاه‌ها: {data.shape[1]}")
        print(f"\n   مقدار ۱۰ ایستگاه اول در زمان اول:")
        print(f"   {data[0, :10]}")
        valid_mask = ~np.isnan(data[0, :10])
        print(f"   تعداد مقادیر معتبر در ۱۰ ایستگاه اول: {np.sum(valid_mask)}")
    elif data.ndim == 3:
        print("   (زمان, ایستگاه, ...) - ۳ بعدی")
        print(f"   تعداد زمان‌ها: {data.shape[0]}")
        print(f"   تعداد ایستگاه‌ها: {data.shape[1]}")
        if data.shape[2] > 1:
            print(f"   بعد سوم: {data.shape[2]}")
        print(f"\n   مقدار ۱۰ ایستگاه اول در زمان اول (و سطح اول):")
        print(f"   {data[0, :10, 0] if data.ndim == 3 else data[0, :10]}")
    else:
        print(f"   بعد غیرمنتظره: {data.ndim} بعدی")
else:
    print(f"\n⚠️ متغیر '{var_name}' در فایل وجود ندارد.")
    print("   متغیرهای موجود:", list(ds.variables.keys()))

# ============================================================
# ۵. شبیه‌سازی خواندن ۱۰ ایستگاه اول
# ============================================================
print("\n" + "=" * 60)
print("🔍 شبیه‌سازی خواندن ۱۰ ایستگاه اول (block_size=10)")
print("=" * 60)

block_start = 0
block_size = 10
station_indices = list(range(block_start, block_start + block_size))

print(f"   اندیس‌های بلوک: {station_indices}")

if station_var:
    # آیا شناسه‌ها از ۰ شروع می‌شوند؟
    first_id = station_ids[0] if len(station_ids) > 0 else None
    if first_id == 0:
        print("   ✅ شناسه‌های ایستگاه از ۰ شروع می‌شوند.")
        exists = all(idx in station_ids for idx in station_indices)
        if exists:
            print("   ✅ ایستگاه‌های ۰ تا ۹ در فایل وجود دارند.")
            if var_name in ds.variables:
                data = ds.variables[var_name][:]
                if data.ndim == 2:
                    extracted = data[:, station_indices]
                elif data.ndim == 3:
                    extracted = data[:, station_indices, :]
                else:
                    extracted = None
                if extracted is not None:
                    print(f"   ✅ داده‌های استخراج‌شده شکل: {extracted.shape}")
                    print(f"   ✅ تعداد مقادیر غیر NaN در کل داده: {np.sum(~np.isnan(extracted))}")
        else:
            print("   ❌ ایستگاه‌های ۰ تا ۹ در فایل وجود ندارند!")
            print(f"   شناسه‌های موجود: {station_ids[:20]}")
    else:
        print(f"   ❌ شناسه‌های ایستگاه از ۰ شروع نمی‌شوند (اولین: {first_id})")
        print(f"   شناسه‌های موجود: {station_ids[:20]}")
        print("   ⚠️ این یعنی باید بین شناسه‌ها و اندیس‌های بلوک نگاشت ایجاد شود.")
        
        # پیدا کردن اندیس‌های واقعی برای ایستگاه‌های ۰ تا ۹
        print("\n   🔧 بررسی: آیا شناسه‌ها از ۱ شروع می‌شوند؟")
        if first_id == 1:
            print("      ✅ شناسه‌ها از ۱ شروع می‌شوند.")
            # اندیس بلوک ۰ معادل شناسه ۱، ۱ معادل ۲ و ...
            mapped_indices = [i + 1 for i in station_indices]
            exists = all(idx in station_ids for idx in mapped_indices)
            if exists:
                print(f"      ✅ ایستگاه‌های شناسه {mapped_indices} در فایل وجود دارند.")
                print("      🔧 راه‌حل: در کد، به‌جای station_indices، از station_indices + 1 استفاده کنید.")
            else:
                print(f"      ❌ ایستگاه‌های شناسه {mapped_indices} وجود ندارند.")
        else:
            print(f"      شناسه‌ها از عدد {first_id} شروع می‌شوند، نیاز به نگاشت دقیق‌تر است.")
else:
    print("   ⚠️ متغیر ایستگاه مشخص نیست، نمی‌توان تطابق را بررسی کرد.")

ds.close()

print("\n" + "=" * 60)
print("✅ تحلیل پایان یافت.")
print("=" * 60)