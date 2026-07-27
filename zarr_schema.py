#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zarr_schema.py
================================================================================
ساختار Zarr برای خروجی نهایی - نسخه ۳.۱
هماهنگ با distribution_registry و پشتیبانی از GEV و Bimodal
================================================================================
"""

import os
import shutil
import numpy as np
import xarray as xr
import zarr
from constants import N_DAYS, N_YEARS, N_VARS, FLOAT_DTYPE, INT_DTYPE

# ============================================================================
# ۱. بارگذاری رجیستری توزیع‌ها
# ============================================================================
try:
    from distribution_registry import DISTRIBUTIONS, get_all_distribution_codes
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("⚠️ distribution_registry یافت نشد. از تعریف دستی استفاده می‌شود.")

# ============================================================================
# ۲. تعریف متغیرهای خروجی بر اساس رجیستری
# ============================================================================

def build_var_defs():
    """
    ساخت لیست تعاریف متغیرها بر اساس DISTRIBUTIONS موجود در registry
    """
    var_defs = []

    if REGISTRY_AVAILABLE:
        for code, info in DISTRIBUTIONS.items():
            dist_name = info["name"].lower()
            # پارامترهای توزیع
            for param in info["params"]:
                pname = param["name"]
                var_name = f"{dist_name}_{pname}"
                var_defs.append((var_name, "float32", np.nan, f"{info['name']} - {pname}"))
            # آماره‌های توزیع (loglik, aicc, bic)
            for stat in ["loglik", "aicc", "bic"]:
                var_name = f"{dist_name}_{stat}"
                var_defs.append((var_name, "float32", np.nan, f"{info['name']} - {stat}"))
    else:
        # تعریف دستی (در صورت نبود registry)
        dists = [
            ("normal", ["p1", "p2"]),
            ("skew", ["p1", "p2", "p3"]),
            ("gev", ["p1", "p2", "p3"]),
            ("bimodal", ["p1", "p2", "p3", "p4", "p5"]),
            ("pearson", ["p1", "p2", "p3"]),
        ]
        for dist_name, params in dists:
            for pname in params:
                var_defs.append((f"{dist_name}_{pname}", "float32", np.nan, f"{dist_name} - {pname}"))
            for stat in ["loglik", "aicc", "bic"]:
                var_defs.append((f"{dist_name}_{stat}", "float32", np.nan, f"{dist_name} - {stat}"))

    # متغیرهای عمومی (همیشه وجود دارند)
    var_defs.append(("best_dist", "int32", -1, "Best distribution index (code)"))
    var_defs.append(("count", "int32", 0, "Number of valid values"))
    var_defs.append(("mean", "float32", np.nan, "Mean"))
    var_defs.append(("std", "float32", np.nan, "Standard deviation"))
    var_defs.append(("skewness", "float32", np.nan, "Skewness"))
    var_defs.append(("median", "float32", np.nan, "Median"))

    return var_defs

# ============================================================================
# ۳. ساختار نهایی
# ============================================================================

VAR_DEFS = build_var_defs()
VAR_NAMES = [v[0] for v in VAR_DEFS]
VAR_DTYPES = {v[0]: v[1] for v in VAR_DEFS}
VAR_FILLS = {v[0]: v[2] for v in VAR_DEFS}
VAR_DESCS = {v[0]: v[3] for v in VAR_DEFS}
N_OUTPUTS = len(VAR_DEFS)

# ============================================================================
# ۴. ایجاد آرایه خالی برای یک بلوک (هم‌جهت با analyze_station)
# ============================================================================
def create_empty_block_result(block_size):
    """
    ایجاد دیکشنری خالی برای ذخیره نتایج یک بلوک
    شکل آرایه‌ها: (N_DAYS, block_size) مطابق با analyze_station
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

# ============================================================================
# ۵. ایجاد Zarr Store (با فشرده‌سازی غیرفعال برای سازگاری)
# ============================================================================
def create_zarr_store(output_path, n_stations, chunk_size=(366, 100)):
    """
    ایجاد یک Zarr Store با ساختار مناسب
    ابعاد: (day_of_year, point)
    """
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    root = zarr.group(output_path, overwrite=True)

    for name, dtype_str, fill, desc in VAR_DEFS:
        if dtype_str == "int32":
            dtype = INT_DTYPE
        else:
            dtype = FLOAT_DTYPE

        root.create_array(
            name,
            shape=(N_DAYS, n_stations),
            chunks=chunk_size,
            dtype=dtype,
            fill_value=fill,
            compressor=None,  # غیرفعال برای سازگاری با Zarr v3
            dimension_names=("day_of_year", "point"),
        )

    # متادیتا
    root.attrs["description"] = "Climatology processing results v3.1"
    root.attrs["n_days"] = N_DAYS
    root.attrs["n_stations"] = n_stations
    root.attrs["variables"] = VAR_NAMES
    root.attrs["variable_descriptions"] = VAR_DESCS
    if REGISTRY_AVAILABLE:
        root.attrs["distributions"] = [info["name"] for info in DISTRIBUTIONS.values()]

    return root

# ============================================================================
# ۶. اضافه کردن مختصات و متادیتا به xarray.Dataset
# ============================================================================
def add_coords_and_metadata(ds, station_ids, lons, lats, elevs):
    """
    اضافه کردن مختصات و متادیتا به دیتاست
    """
    n_stations = len(station_ids)

    ds = ds.assign_coords(point=np.arange(n_stations))
    ds["stationid"] = ("point", station_ids)
    ds["lon"] = ("point", lons)
    ds["lat"] = ("point", lats)
    ds["elev"] = ("point", elevs)

    ds.attrs["distributions"] = (
        str([info["name"] for info in DISTRIBUTIONS.values()])
        if REGISTRY_AVAILABLE
        else "unknown"
    )
    ds.attrs["variables"] = VAR_NAMES
    ds.attrs["n_outputs"] = N_OUTPUTS

    return ds

# ============================================================================
# ۷. اجرای آزمایشی
# ============================================================================
if __name__ == "__main__":
    print(f"✅ {N_OUTPUTS} متغیر برای {len(VAR_DEFS)} تعریف شد.")
    print(f"   نمونه متغیرها: {VAR_NAMES[:8]}...")