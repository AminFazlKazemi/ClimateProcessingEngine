#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_read_one_point.py
خواندن یک نقطه از یک فایل Zarr و نمایش مقادیر خام و اسکالشده
"""

import os
import numpy as np
import xarray as xr

# ============================================================
# تنظیمات
# ============================================================
ZARR_FILE = r"K:\gozareshha\dr vazife\140504 - qc temp\zarr_yearly_monthly\1369_04_Tir.zarr"
POINT_INDEX = 0  # اولین ایستگاه

# ============================================================
# ۱. باز کردن فایل Zarr
# ============================================================
print("=" * 60)
print("📂 باز کردن فایل:", ZARR_FILE)
ds = xr.open_zarr(ZARR_FILE, consolidated=False)

print("📌 ابعاد:", dict(ds.dims))
print("📌 متغیرها:", list(ds.data_vars))

# ============================================================
# ۲. خواندن داده‌های نقطه اول برای هر سه متغیر
# ============================================================
print("\n" + "=" * 60)
print(f"📊 داده‌های نقطه {POINT_INDEX}:")

for var_name in ['tmin', 'tmean', 'tmax']:
    if var_name not in ds:
        print(f"⚠️ متغیر {var_name} وجود ندارد.")
        continue

    # استخراج داده‌های نقطه اول
    data = ds[var_name].isel(point=POINT_INDEX).values
    print(f"\n🔹 {var_name.upper()}:")
    print(f"   نوع داده: {data.dtype}")
    print(f"   تعداد روزها: {len(data)}")
    print(f"   مقدار min: {np.min(data)}")
    print(f"   مقدار max: {np.max(data)}")
    print(f"   مقدار mean: {np.mean(data):.2f}")
    print(f"   مقدار std:  {np.std(data):.2f}")

    # ۱۰ روز اول
    print(f"   ۱۰ روز اول: {data[:10]}")

    # بررسی اینکه آیا داده باید اسکیل شود
    if np.max(data) > 100:
        print(f"   ⚠️ محدوده‌ی بزرگ ({np.max(data)}) نشان‌دهنده‌ی نیاز به اسکیل (÷۱۰) است.")
        scaled = data.astype(np.float32) / 10.0
        print(f"   ✅ پس از تقسیم بر ۱۰: min={np.min(scaled):.2f}, max={np.max(scaled):.2f}")
        print(f"   ۱۰ روز اول اسکالشده: {scaled[:10]}")
    else:
        print(f"   ✅ داده به نظر در مقیاس درست است (بدون نیاز به اسکیل).")

ds.close()

# ============================================================
# ۳. تست با تابع get_cached_or_load (اگر موجود باشد)
# ============================================================
print("\n" + "=" * 60)
print("🧪 تست تابع get_cached_or_load (اگر قابل import باشد):")

try:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from io_pipeline.read_month_files import get_cached_or_load

    # ساختار fake برای تست
    year = 1369
    month = 4
    var_idx = 0  # tmin
    block_start = 0
    block_size = 1
    zarr_path = ZARR_FILE

    data_from_func = get_cached_or_load(year, month, var_idx, block_start, block_size, zarr_path)
    if data_from_func is not None:
        print(f"   داده از get_cached_or_load برای tmin: {data_from_func.shape}")
        print(f"   min: {np.min(data_from_func)}, max: {np.max(data_from_func)}")
        print(f"   ۱۰ روز اول: {data_from_func[:10].flatten()}")
    else:
        print("   ❌ تابع داده‌ای برنگرداند.")

except ImportError as e:
    print(f"   ❌ خطا در import: {e}")
except Exception as e:
    print(f"   ❌ خطا در اجرا: {e}")

print("\n" + "=" * 60)
print("✅ بررسی کامل شد.")