#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_first_blocks.py – بررسی کامل بودن بلوک‌های ۰ تا ۳
"""

import os
import numpy as np
import xarray as xr

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000
NUM_BLOCKS = 4  # بلوک‌های ۰ تا ۳

print("=" * 70)
print("🔍 بررسی کامل بودن بلوک‌های ۰ تا ۳")
print("=" * 70)

ds = xr.open_zarr(ZARR_PATH, consolidated=False)

for block_idx in range(NUM_BLOCKS):
    block_start = block_idx * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    print(f"\n📦 بلوک {block_idx}: ایستگاه‌های {block_start:,} تا {block_end-1:,}")
    
    all_complete = True
    for var in ['tmin', 'tmean', 'tmax']:
        data = ds[f'{var}_mean'].isel(day=0).values[block_start:block_end]
        valid = np.sum(~np.isnan(data))
        if valid == BLOCK_SIZE:
            print(f"   {var.upper()}: ✅ کامل ({valid}/{BLOCK_SIZE})")
        else:
            print(f"   {var.upper()}: ⚠️ ناقص ({valid}/{BLOCK_SIZE})")
            all_complete = False
    
    if all_complete:
        print("   ✅ این بلوک کامل است.")
    else:
        print("   ❌ این بلوک ناقص است.")

ds.close()
print("\n" + "=" * 70)