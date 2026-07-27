#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zarr_schema.py
============================================
ساختار Zarr برای خروجی نهایی
شامل: ایجاد store، اضافه کردن مختصات و متادیتا
"""

import os
import shutil
import numpy as np
import xarray as xr
import zarr
from constants import (
    N_DAYS, N_YEARS, N_VARS, FLOAT_DTYPE, INT_DTYPE,
    DISTRIBUTIONS, MAX_VALUES_PER_FIT
)

# ============================================================
# تعریف متغیرهای خروجی
# ============================================================
VAR_DEFS = []

for dist in DISTRIBUTIONS:
    dist_name = dist["name"]
    # پارامترهای توزیع
    for pname, dtype, desc in dist["params"]:
        var_name = f"{dist_name}_{pname}"
        VAR_DEFS.append((var_name, dtype, -1, desc))
    # آماره‌های توزیع
    for stat in ["loglik", "aicc", "bic"]:
        var_name = f"{dist_name}_{stat}"
        VAR_DEFS.append((var_name, "float32", np.nan, f"{stat} for {dist_name}"))

# متغیر best_dist (شاخص بهترین توزیع)
VAR_DEFS.append(("best_dist", "int32", -1, "Best distribution index"))

# متغیر count (تعداد مقادیر معتبر)
VAR_DEFS.append(("count", "int32", 0, "Number of valid values"))

# متغیر std (انحراف معیار)
VAR_DEFS.append(("std", "float32", np.nan, "Standard deviation"))

# نام متغیرها و نوع داده‌ها
VAR_NAMES = [v[0] for v in VAR_DEFS]
VAR_DTYPES = {v[0]: v[1] for v in VAR_DEFS}
VAR_FILLS = {v[0]: v[2] for v in VAR_DEFS}
VAR_DESCS = {v[0]: v[3] for v in VAR_DEFS}

# تعداد کل متغیرهای خروجی
N_OUTPUTS = len(VAR_DEFS)

# ============================================================
# ایجاد آرایه خالی برای یک بلوک
# ============================================================
def create_empty_block_result(block_size):
    """
    ایجاد یک دیکشنری خالی برای ذخیره نتایج یک بلوک
    
    Args:
        block_size: تعداد ایستگاه‌ها در بلوک
    
    Returns:
        dict: دیکشنری با کلیدهای VAR_NAMES و آرایه‌های خالی با شکل (N_DAYS, block_size)
    """
    block_result = {}
    for name in VAR_NAMES:
        dtype = VAR_DTYPES[name]
        fill = VAR_FILLS[name]
        if dtype == "int32":
            arr = np.full((N_DAYS, block_size), fill, dtype=INT_DTYPE)
        else:
            arr = np.full((N_DAYS, block_size), fill, dtype=FLOAT_DTYPE)
        block_result[name] = arr
    return block_result

# ============================================================
# ایجاد Zarr Store
# ============================================================
def create_zarr_store(output_path, n_stations, chunk_size=(366, 100)):
    """
    ایجاد یک Zarr Store با ساختار مناسب
    
    Args:
        output_path: مسیر ذخیره Zarr
        n_stations: تعداد ایستگاه‌ها
        chunk_size: اندازه تکه‌ها (day_of_year, point)
    
    Returns:
        zarr.Group: گروه Zarr
    """
    # حذف پوشه قبلی در صورت وجود
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # ایجاد گروه Zarr
    root = zarr.group(output_path, overwrite=True)
    
    # ایجاد هر متغیر به‌صورت آرایه بدون فشرده‌سازی (برای سازگاری با Zarr v3)
    for name, dtype_str, fill, desc in VAR_DEFS:
        # تبدیل نوع داده به numpy dtype
        if dtype_str == "int32":
            dtype = INT_DTYPE
        else:
            dtype = FLOAT_DTYPE
        
        # ایجاد آرایه بدون compressor (سازگار با Zarr v3)
        root.create_array(
            name,
            shape=(N_DAYS, n_stations),
            chunks=chunk_size,
            dtype=dtype,
            fill_value=fill,
            compressor=None,  # ⬅️ فشرده‌سازی غیرفعال برای سازگاری
            dimension_names=("day_of_year", "point"),
        )
    
    # اضافه کردن متادیتا
    root.attrs["description"] = "Climatology processing results"
    root.attrs["n_days"] = N_DAYS
    root.attrs["n_stations"] = n_stations
    root.attrs["variables"] = VAR_NAMES
    root.attrs["variable_descriptions"] = VAR_DESCS
    
    return root

# ============================================================
# اضافه کردن مختصات و متادیتا
# ============================================================
def add_coords_and_metadata(ds, station_ids, lons, lats, elevs):
    """
    اضافه کردن مختصات و متادیتا به دیتاست
    
    Args:
        ds: xarray.Dataset
        station_ids: شناسه‌های ایستگاه‌ها
        lons: طول‌های جغرافیایی
        lats: عرض‌های جغرافیایی
        elevs: ارتفاع‌ها
    
    Returns:
        xarray.Dataset: دیتاست با مختصات و متادیتا
    """
    n_stations = len(station_ids)
    
    # اضافه کردن مختصات
    ds = ds.assign_coords(
        point=np.arange(n_stations),
    )
    
    # اضافه کردن متغیرهای مختصات
    ds["stationid"] = ("point", station_ids)
    ds["lon"] = ("point", lons)
    ds["lat"] = ("point", lats)
    ds["elev"] = ("point", elevs)
    
    # اضافه کردن متادیتا
    ds.attrs["distributions"] = str([d["name"] for d in DISTRIBUTIONS])
    ds.attrs["variables"] = VAR_NAMES
    ds.attrs["n_outputs"] = N_OUTPUTS
    
    return ds

# ============================================================
# اجرای آزمایشی
# ============================================================
if __name__ == "__main__":
    print(f"✅ {N_OUTPUTS} متغیر برای {len(DISTRIBUTIONS)} توزیع")
    print(f"   متغیرها: {VAR_NAMES[:5]}...")