#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_scale_fix.py
تست مستقیم توابع بارگذاری برای بررسی اعمال اسکیل
"""

import os
import sys
import numpy as np

# اضافه کردن مسیر پروژه به sys.path
BASE_DIR = r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine"
sys.path.insert(0, BASE_DIR)

from constants import ZARR_BASE, VARS, CACHE_DIR
from runtime_tables import build_runtime_tables
from io_pipeline.read_month_files import get_cached_or_load, read_month_files, clear_ds_cache

# ============================================================
# تنظیمات
# ============================================================
YEAR = 1369
MONTH = 4  # تیر
POINT_INDEX = 0
BLOCK_START = 0
BLOCK_SIZE = 1  # فقط یک ایستگاه

print("=" * 60)
print("🧪 تست اسکیل داده‌ها با توابع واقعی")
print("=" * 60)
print(f"📂 ZARR_BASE: {ZARR_BASE}")
print(f"📅 سال: {YEAR}, ماه: {MONTH}")
print(f"📍 نقطه: {POINT_INDEX}")
print("=" * 60)

# ============================================================
# ۱. ساخت file_map
# ============================================================
print("\n📋 ساخت file_map...")
tables = build_runtime_tables(ZARR_BASE)
file_map = tables["file_map"]
year_list = [YEAR]
print(f"   ✅ تعداد فایل‌ها در file_map: {len(file_map)}")

# ============================================================
# ۲. تست تابع get_cached_or_load (تک‌متغیره)
# ============================================================
print("\n" + "=" * 60)
print("🧪 تست تابع get_cached_or_load")
print("=" * 60)

for var_idx, var_name in enumerate(VARS):
    print(f"\n🔹 متغیر: {var_name}")
    try:
        data = get_cached_or_load(
            year=YEAR,
            month=MONTH,
            var_idx=var_idx,
            block_start=BLOCK_START,
            block_size=BLOCK_SIZE,
            zarr_path=None  # از ZARR_BASE استفاده می‌کند
        )
        if data is None:
            print(f"   ❌ داده‌ای برگردانده نشد.")
            continue

        print(f"   نوع داده: {data.dtype}")
        print(f"   شکل: {data.shape}")
        print(f"   min: {np.min(data):.2f}")
        print(f"   max: {np.max(data):.2f}")
        print(f"   mean: {np.mean(data):.2f}")
        print(f"   ۱۰ روز اول (نقطه ۰): {data[:10, 0] if data.ndim > 1 else data[:10]}")

        # بررسی اسکیل
        if np.max(data) > 100:
            print(f"   ⚠️ محدوده‌ی بزرگ ({np.max(data):.0f}) نشان می‌دهد که اسکیل اعمال نشده است!")
        else:
            print(f"   ✅ محدوده‌ی منطقی است. اسکیل به درستی اعمال شده است.")

    except Exception as e:
        print(f"   ❌ خطا: {e}")

# ============================================================
# ۳. تست تابع read_month_files (همه متغیرها)
# ============================================================
print("\n" + "=" * 60)
print("🧪 تست تابع read_month_files (همه متغیرها)")
print("=" * 60)

try:
    data_dict = read_month_files(
        block_start=BLOCK_START,
        block_size=BLOCK_SIZE,
        file_map=file_map,
        year_list=year_list,
        var_idx=None  # همه متغیرها
    )

    if not data_dict:
        print("❌ داده‌ای برگردانده نشد.")
    else:
        for key, data in data_dict.items():
            print(f"\n📂 کلید: {key}")
            print(f"   شکل: {data.shape}")
            print(f"   نوع داده: {data.dtype}")

            # داده‌ها به صورت (days, points, n_vars) هستند
            if data.ndim == 3:
                days, points, n_vars = data.shape
                for v, var_name in enumerate(VARS):
                    if v < n_vars:
                        var_data = data[:, 0, v]  # نقطه ۰
                        print(f"\n   🔹 {var_name}:")
                        print(f"      min: {np.min(var_data):.2f}")
                        print(f"      max: {np.max(var_data):.2f}")
                        print(f"      mean: {np.mean(var_data):.2f}")
                        print(f"      ۱۰ روز اول: {var_data[:10]}")

                        # بررسی اسکیل
                        if np.max(var_data) > 100:
                            print(f"      ⚠️ محدوده‌ی بزرگ ({np.max(var_data):.0f}) نشان می‌دهد که اسکیل اعمال نشده است!")
                        else:
                            print(f"      ✅ محدوده‌ی منطقی است. اسکیل به درستی اعمال شده است.")
            else:
                print(f"   شکل غیرمنتظره: {data.shape}")

except Exception as e:
    print(f"❌ خطا در read_month_files: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# ۴. پاک کردن کش (برای تست‌های بعدی)
# ============================================================
print("\n" + "=" * 60)
print("🗑️ پاک کردن کش...")
clear_ds_cache()
print("✅ کش پاک شد.")

print("\n" + "=" * 60)
print("✅ تست کامل شد.")
print("=" * 60)