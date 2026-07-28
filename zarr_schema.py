# -*- coding: utf-8 -*-
"""
zarr_schema.py - تعریف ساختار خروجی Zarr (با ایجاد مستقیم روی دیسک و پشتیبانی از ادامه پردازش)
"""

import os
import shutil
import numpy as np
import zarr
from numcodecs import Blosc
from constants import N_DAYS, FLOAT_DTYPE, INT_DTYPE

# ============================================================
# تعریف توزیع‌ها
# ============================================================
DISTRIBUTIONS = {
    0: {"name": "Normal", "params": ["mean", "std"], "n_params": 2},
    1: {"name": "SkewNormal", "params": ["alpha", "loc", "scale"], "n_params": 3},
    2: {"name": "Bimodal", "params": ["w1", "mu1", "sigma1", "mu2", "sigma2"], "n_params": 5},
    3: {"name": "Pearson", "params": ["shape", "scale", "loc"], "n_params": 3},
}

# ============================================================
# ساخت لیست کامل متغیرها
# ============================================================
VAR_NAMES = []
VAR_DTYPES = {}

for var in ['tmin', 'tmean', 'tmax']:
    for suffix in ['best_dist', 'mean', 'std', 'skewness', 'median', 'count']:
        name = f"{var}_{suffix}"
        VAR_NAMES.append(name)
        if suffix in ['best_dist', 'count']:
            VAR_DTYPES[name] = INT_DTYPE
        else:
            VAR_DTYPES[name] = FLOAT_DTYPE

for var in ['tmin', 'tmean', 'tmax']:
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
            result[name] = np.full((N_DAYS, block_size), -1, dtype=dtype)
        else:
            result[name] = np.full((N_DAYS, block_size), np.nan, dtype=dtype)
    return result

def create_zarr_store(output_path, n_stations):
    """
    ایجاد فروشگاه Zarr روی دیسک بدون بارگذاری داده در حافظه.
    هر آرایه به‌صورت خالی با fill_value ایجاد می‌شود.
    """
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"   🗑️ Removed existing store")

    root = zarr.open_group(output_path, mode='w', zarr_format=2)

    compressor = Blosc(cname='zstd', clevel=3, shuffle=2)

    print(f"   📊 Creating {len(VAR_NAMES)} variables directly on disk...")
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
        # متادیتای مورد نیاز xarray
        arr.attrs['_ARRAY_DIMENSIONS'] = ['day', 'point']

        if (idx + 1) % 10 == 0 or idx == len(VAR_NAMES) - 1:
            print(f"      {idx+1}/{len(VAR_NAMES)} variables created")

    root.attrs['zarr_version'] = 2
    print(f"   ✅ Zarr store created successfully at: {output_path}")
    print(f"   📊 Total variables: {len(VAR_NAMES)}")
    print(f"   📊 Shape: ({N_DAYS}, {n_stations})")
    print(f"   📊 Chunks: ({chunks[0]}, {chunks[1]})")

    return root

def get_or_create_zarr_store(output_path, n_stations):
    """
    اگر Zarr وجود داشته باشد، آن را باز می‌کند (حالت append).
    در غیر این صورت، یک Zarr جدید ایجاد می‌کند.
    """
    if os.path.exists(output_path):
        print(f"   📂 Zarr exists, opening it: {output_path}")
        root = zarr.open_group(output_path, mode='a')

        # بررسی متغیرهای موجود
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
                print(f"      Added missing variable: {name}")

        # بررسی ابعاد
        sample_name = next(iter(existing_vars)) if existing_vars else None
        if sample_name:
            existing_shape = root[sample_name].shape
            expected_shape = (N_DAYS, n_stations)
            if existing_shape != expected_shape:
                print(f"   ⚠️ Shape mismatch: existing {existing_shape} != expected {expected_shape}")
                print(f"   🔄 Recreating store with correct shape...")
                return create_zarr_store(output_path, n_stations)

        return root
    else:
        print(f"   🆕 Zarr does not exist, creating new: {output_path}")
        return create_zarr_store(output_path, n_stations)

def add_coords_and_metadata(ds, station_ids, lons, lats, elevs):
    """افزودن مختصات و متادیتا به دیتاست xarray"""
    ds = ds.assign_coords(
        stationid=("point", station_ids),
        lon=("point", lons),
        lat=("point", lats),
        elev=("point", elevs)
    )
    return ds