# -*- coding: utf-8 -*-
"""
zarr_schema.py - تعریف ساختار خروجی Zarr (نسخه هوشمند)
"""

import os
import shutil
import numpy as np
import zarr
from numcodecs import Blosc
from constants import N_DAYS, FLOAT_DTYPE, INT_DTYPE

# ============================================================
# تعریف توزیع‌ها (هماهنگ با distributions.py)
# ============================================================
DISTRIBUTIONS = {
    0: {"name": "Normal", "params": ["mean", "std"], "n_params": 2},
    1: {"name": "Skew", "params": ["alpha", "loc", "scale"], "n_params": 3},
    2: {"name": "GEV", "params": ["shape", "loc", "scale"], "n_params": 3},
    3: {"name": "Bimodal", "params": ["w1", "mu1", "sigma1", "mu2", "sigma2"], "n_params": 5},
    4: {"name": "Pearson", "params": ["shape", "scale", "loc"], "n_params": 3},
}

# ============================================================
# ساخت لیست کامل متغیرها
# ============================================================
VAR_NAMES = []
VAR_DTYPES = {}

for var in ['tmin', 'tmean', 'tmax']:
    # آماره‌های پایه
    for suffix in ['best_dist', 'mean', 'std', 'skewness', 'median', 'count']:
        name = f"{var}_{suffix}"
        VAR_NAMES.append(name)
        VAR_DTYPES[name] = INT_DTYPE if suffix in ['best_dist', 'count'] else FLOAT_DTYPE

    # پارامترها و معیارهای اطلاعاتی هر توزیع
    for dist_info in DISTRIBUTIONS.values():
        dist_name = dist_info["name"].lower()
        for i in range(1, dist_info["n_params"] + 1):
            name = f"{var}_{dist_name}_p{i}"
            VAR_NAMES.append(name)
            VAR_DTYPES[name] = FLOAT_DTYPE
        for suffix in ['loglik', 'aicc', 'bic']:
            name = f"{var}_{dist_name}_{suffix}"
            VAR_NAMES.append(name)
            VAR_DTYPES[name] = FLOAT_DTYPE

# ============================================================
# توابع کمکی
# ============================================================

def create_empty_block_result(block_size):
    """ایجاد دیکشنری خالی برای نتایج یک بلوک"""
    result = {}
    for name in VAR_NAMES:
        dtype = VAR_DTYPES[name]
        if dtype == INT_DTYPE:
            if name.endswith('_count'):
                result[name] = np.zeros((N_DAYS, block_size), dtype=dtype)
            else:
                result[name] = np.full((N_DAYS, block_size), -1, dtype=dtype)
        else:
            result[name] = np.full((N_DAYS, block_size), np.nan, dtype=dtype)
    return result

def create_zarr_store(output_path, n_stations):
    """ایجاد فروشگاه Zarr روی دیسک"""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"   🗑️ Removed existing store")

    root = zarr.open_group(output_path, mode='w', zarr_format=2)
    compressor = Blosc(cname='zstd', clevel=3, shuffle=2)

    print(f"   📊 Creating {len(VAR_NAMES)} variables...")
    for idx, name in enumerate(VAR_NAMES):
        dtype = VAR_DTYPES[name]
        fill_value = -1 if dtype == INT_DTYPE else np.nan
        chunks = (min(N_DAYS, 366), min(500, n_stations))

        arr = root.create(
            name,
            shape=(N_DAYS, n_stations),
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            compressor=compressor,
            overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['day', 'point']

        if (idx + 1) % 10 == 0 or idx == len(VAR_NAMES) - 1:
            print(f"      {idx+1}/{len(VAR_NAMES)} variables created")

    root.attrs['zarr_version'] = 2
    print(f"   ✅ Zarr store created: {output_path}")
    return root

def get_or_create_zarr_store(output_path, n_stations):
    """
    باز کردن یا ایجاد فروشگاه Zarr با مدیریت هوشمند ابعاد.
    - متغیرهای گم‌شده را اضافه می‌کند.
    - فقط متغیرهای دو بعدی را برای بررسی ابعاد در نظر می‌گیرد.
    - اگر متغیر یک‌بعدی (مختصات) وجود داشته باشد، نادیده گرفته می‌شود.
    """
    if not os.path.exists(output_path):
        return create_zarr_store(output_path, n_stations)

    print(f"   📂 Zarr exists, opening it: {output_path}")
    root = zarr.open_group(output_path, mode='a')

    # ============================================================
    # ۱. پیدا کردن یک متغیر دو بعدی نمونه برای بررسی ابعاد
    # ============================================================
    sample_var = None
    for name in root.array_keys():
        arr = root[name]
        if len(arr.shape) == 2 and arr.shape[0] == N_DAYS:
            sample_var = name
            break

    if sample_var is None:
        # اگر هیچ متغیر دو بعدی با بعد روز وجود ندارد، احتمالاً Zarr خراب است یا خالی
        print(f"   ⚠️ No valid 2D variable found. Recreating store...")
        root = create_zarr_store(output_path, n_stations)
        return root

    # ============================================================
    # ۲. بررسی ابعاد
    # ============================================================
    existing_shape = root[sample_var].shape
    expected_shape = (N_DAYS, n_stations)

    if existing_shape != expected_shape:
        print(f"   ⚠️ Shape mismatch: existing {existing_shape} != expected {expected_shape}")
        print(f"   ⚠️ Continuing with existing shape ({existing_shape[1]} points).")
        print(f"   ⚠️ To change shape, delete the Zarr store manually and re-run.")
        # ابعاد را تغییر نمی‌دهیم، فقط هشدار می‌دهیم

    # ============================================================
    # ۳. اضافه کردن متغیرهای گم‌شده
    # ============================================================
    existing_vars = set(root.array_keys())
    expected_vars = set(VAR_NAMES)
    missing = expected_vars - existing_vars

    if missing:
        print(f"   ⚠️ Missing variables: {missing}. Adding them...")
        compressor = Blosc(cname='zstd', clevel=3, shuffle=2)
        for name in missing:
            dtype = VAR_DTYPES[name]
            fill_value = -1 if dtype == INT_DTYPE else np.nan
            chunks = (min(N_DAYS, 366), min(500, n_stations))
            arr = root.create(
                name,
                shape=(N_DAYS, n_stations),
                chunks=chunks,
                dtype=dtype,
                fill_value=fill_value,
                compressor=compressor,
                overwrite=True
            )
            arr.attrs['_ARRAY_DIMENSIONS'] = ['day', 'point']
            print(f"      Added: {name}")

    return root

def add_coords_and_metadata(ds, station_ids, lons, lats, elevs):
    """افزودن مختصات و متادیتا به دیتاست xarray"""
    ds = ds.assign_coords(
        stationid=("point", station_ids),
        lon=("point", lons),
        lat=("point", lats),
        elev=("point", elevs)
    )
    return ds