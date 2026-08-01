#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_progress.py – بررسی تعداد نقاط پردازش‌شده در Zarr
================================================================================
- تعداد ایستگاه‌های معتبر را در روز اول (یا روز مشخص) محاسبه می‌کند.
- پیشرفت کلی و تعداد بلوک‌های تکمیل‌شده را نمایش می‌دهد.
================================================================================
"""

import numpy as np
import xarray as xr
from pathlib import Path

# ============================================================
# مسیر فایل Zarr خروجی
# ============================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000  # مطابق config.yaml


def check_progress(var_name="tmean_mean", day_idx=0):
    """بررسی تعداد نقاط پردازش‌شده بر اساس یک متغیر مشخص در یک روز خاص."""
    if not Path(ZARR_PATH).exists():
        print(f"❌ فایل Zarr در مسیر {ZARR_PATH} وجود ندارد!")
        return

    print("📂 باز کردن فایل Zarr...")
    ds = xr.open_zarr(ZARR_PATH, consolidated=False)

    if var_name not in ds:
        print(f"❌ متغیر {var_name} در Zarr وجود ندارد.")
        print(f"   متغیرهای موجود: {list(ds.data_vars)[:10]} ...")
        ds.close()
        return

    # ✅ استفاده از ds.sizes به جای ds.dims (رفع FutureWarning)
    n_days = ds.sizes["day"]
    n_stations = ds.sizes["point"]
    print(f"📊 ابعاد: {n_days} روز × {n_stations:,} ایستگاه")

    data = ds[var_name].isel(day=day_idx).values
    valid_mask = ~np.isnan(data)
    processed_stations = np.sum(valid_mask)

    total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
    processed_blocks = processed_stations // BLOCK_SIZE
    percentage = (processed_stations / n_stations) * 100

    print("\n" + "=" * 60)
    print(f"📊 وضعیت پردازش (بر اساس {var_name} در روز {day_idx + 1})")
    print("=" * 60)
    print(f"   ✅ ایستگاه‌های پردازش‌شده:  {processed_stations:,} از {n_stations:,}")
    print(f"   📈 درصد پیشرفت:             {percentage:.2f}%")
    print(f"   📦 اندازه‌ی بلوک:            {BLOCK_SIZE}")
    print(f"   📦 تعداد بلوک‌های کامل:      {processed_blocks} از {total_blocks}")
    print(f"   ⏳ ایستگاه‌های باقی‌مانده:  {n_stations - processed_stations:,}")

    if day_idx == 0 and processed_stations > 0:
        last_day_data = ds[var_name].isel(day=-1).values
        last_day_valid = np.sum(~np.isnan(last_day_data))
        print(f"\n📊 مقایسه با آخرین روز (روز {n_days}):")
        print(f"   ✅ ایستگاه‌های معتبر در آخرین روز: {last_day_valid:,}")
        if last_day_valid < processed_stations:
            print("   ⚠️ آخرین روز کمتر از روز اول است → احتمالاً پردازش در حال انجام است.")
        elif last_day_valid == processed_stations:
            print("   ✅ روز آخر برابر با روز اول است → پردازش کامل شده یا متوقف شده.")

    ds.close()
    print("=" * 60)


def main():
    check_progress("tmean_mean", day_idx=0)
    print("\n")
    check_progress("tmean_mean", day_idx=-1)


if __name__ == "__main__":
    main()  # ← حتماً این خط را بدون کاراکتر اضافی بنویسید