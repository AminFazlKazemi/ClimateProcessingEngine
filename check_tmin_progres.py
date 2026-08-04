#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_tmin_progress.py - بررسی وضعیت پردازش tmin
"""

import xarray as xr
import numpy as np

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000

print("📂 باز کردن Zarr...")
ds = xr.open_zarr(ZARR_PATH, consolidated=False)

n_stations = ds.sizes['point']
print(f"📍 تعداد کل نقاط: {n_stations:,}\n")

# متغیر tmin
var = 'tmin'

# روز اول
data0 = ds[f'{var}_mean'].isel(day=0).values
valid0 = ~np.isnan(data0)
processed0 = np.sum(valid0)
pct0 = processed0 / n_stations * 100

# آخرین روز
data_last = ds[f'{var}_mean'].isel(day=-1).values
valid_last = ~np.isnan(data_last)
processed_last = np.sum(valid_last)
pct_last = processed_last / n_stations * 100

print(f"🔹 {var.upper()}:")
print(f"   روز اول   : {processed0:,} / {n_stations:,} ({pct0:.2f}%)")
print(f"   آخرین روز : {processed_last:,} / {n_stations:,} ({pct_last:.2f}%)")

# بلوک‌های کامل بر اساس روز اول
blocks_complete = processed0 // BLOCK_SIZE
total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
print(f"   بلوک‌های کامل: {blocks_complete} از {total_blocks}")

# نقطه آخرین معتبر (بر اساس روز اول)
if processed0 > 0:
    last_valid_idx = np.max(np.where(valid0)[0])
    print(f"   آخرین نقطه معتبر (روز اول): {last_valid_idx:,}")

ds.close()