#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resize_zarr.py – تغییر ابعاد فایل Zarr از ۴۰,۰۰۰ به ۳۳۸,۶۲۷ نقطه
داده‌های قبلی حفظ می‌شوند و نقاط جدید با NaN پر می‌شوند.
"""

import zarr
from zarr_schema import VAR_NAMES

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
NEW_N_POINTS = 338627

print(f"📂 باز کردن فایل Zarr: {ZARR_PATH}")
root = zarr.open(ZARR_PATH, mode="a")

print(f"📊 ابعاد فعلی: (366, {root[VAR_NAMES[0]].shape[1]})")
print(f"📊 ابعاد جدید: (366, {NEW_N_POINTS})")

for name in VAR_NAMES:
    arr = root[name]
    old_shape = arr.shape
    new_shape = (old_shape[0], NEW_N_POINTS)
    print(f"   🔄 در حال resize: {name} ...")
    arr.resize(new_shape)

print("✅ همه‌ی متغیرها با موفقیت resize شدند.")
print(f"📁 فایل Zarr اکنون ابعاد (366, {NEW_N_POINTS}) دارد.")