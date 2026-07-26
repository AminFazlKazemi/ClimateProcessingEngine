#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zarr_schema.py - شِمای پویا با پشتیبانی از توزیع‌های متعدد
"""

import os
import numpy as np
import zarr
import zarr.codecs
from constants import N_DAYS, N_YEARS, FLOAT_DTYPE, INT_DTYPE

# ============================================================================
# تعریف توزیع‌ها
# ============================================================================
DISTRIBUTIONS = [
    {
        "name": "normal",
        "params": [("p1", "f4", "mean"), ("p2", "f4", "std")],
        "code": 0,
        "param_count": 2,
    },
    {
        "name": "skew",
        "params": [("p1", "f4", "alpha"), ("p2", "f4", "loc"), ("p3", "f4", "scale")],
        "code": 1,
        "param_count": 3,
    },
    {
        "name": "bimodal",
        "params": [("p1", "f4", "w1"), ("p2", "f4", "mu1"), ("p3", "f4", "sigma1"),
                   ("p4", "f4", "mu2"), ("p5", "f4", "sigma2")],
        "code": 2,
        "param_count": 5,
    },
    {
        "name": "pearson",
        "params": [("p1", "f4", "shape"), ("p2", "f4", "scale"), ("p3", "f4", "loc")],
        "code": 3,
        "param_count": 3,
    },
]

# ساخت خودکار متغیرها
VAR_DEFS = [
    ("best_dist", "i4", -1, "کد بهترین توزیع"),
    ("mean", "f4", np.nan, "میانگین"),
    ("std", "f4", np.nan, "انحراف معیار"),
    ("skewness", "f4", np.nan, "چولگی"),
    ("median", "f4", np.nan, "میانه"),
    ("count", "i4", 0, "تعداد داده‌های معتبر"),
]

for dist in DISTRIBUTIONS:
    name = dist["name"]
    for pname, dtype, desc in dist["params"]:
        VAR_DEFS.append((f"{name}_{pname}", dtype, np.nan, desc))
    for stat in ["loglik", "aicc", "bic"]:
        VAR_DEFS.append((f"{name}_{stat}", "f4", np.nan, f"{stat.upper()}"))

N_OUTPUTS = len(VAR_DEFS)
VAR_NAMES = [v[0] for v in VAR_DEFS]
VAR_DTYPES = {v[0]: v[1] for v in VAR_DEFS}
VAR_FILLS = {v[0]: v[2] for v in VAR_DEFS}
VAR_DESCS = {v[0]: v[3] for v in VAR_DEFS}

def get_dtype(dtype_str):
    return INT_DTYPE if dtype_str == "i4" else FLOAT_DTYPE

def create_empty_block_result(block_size):
    result = {}
    for name, dtype_str, fill, _ in VAR_DEFS:
        dtype = get_dtype(dtype_str)
        result[name] = np.full((N_DAYS, block_size), fill, dtype=dtype)
    return result

def create_zarr_store(output_path, n_stations, chunk_size=(366, 100)):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    root = zarr.open(output_path, mode="w")
    blosc = zarr.codecs.Blosc(cname="zstd", clevel=3)
    for name, dtype_str, fill, desc in VAR_DEFS:
        dtype = get_dtype(dtype_str)
        root.create_array(
            name,
            shape=(N_DAYS, n_stations),
            dtype=dtype,
            chunks=chunk_size,
            fill_value=fill,
            compressors=[blosc],
            dimension_names=("day_of_year", "point"),
        )
    return root

def add_coords_and_metadata(ds, station_ids, lons, lats, elevs):
    ds = ds.assign_coords({
        "day_of_year": np.arange(1, N_DAYS + 1),
        "point": np.arange(len(station_ids)),
        "stationid": ("point", station_ids),
        "lon": ("point", lons),
        "lat": ("point", lats),
        "elev": ("point", elevs),
    })
    ds.attrs["description"] = "Climatology with dynamic distributions"
    ds.attrs["distributions"] = str([d["name"] for d in DISTRIBUTIONS])
    ds.attrs["n_outputs"] = N_OUTPUTS
    return ds

if __name__ == "__main__":
    print(f"✅ {N_OUTPUTS} متغیر برای {len(DISTRIBUTIONS)} توزیع")
