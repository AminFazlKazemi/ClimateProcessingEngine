#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_pipeline/read_month_files.py
================================================================================
خواندن فایل‌های ماهانه Zarr.
فقط خواندن. هیچ rearranging انجام نمی‌شود.
================================================================================
ورژن: 2.0 - نهایی
"""

import numpy as np
import xarray as xr
from constants import VARS

def read_month_files(block_start, block_size, file_map, year_list):
    data_dict = {}
    for year_idx, year in enumerate(year_list):
        for month in range(1, 13):
            key = (year, month)
            if key not in file_map:
                continue
            ds = xr.open_zarr(file_map[key], consolidated=False)
            ds_block = ds.isel(point=slice(block_start, block_start + block_size))
            for var_idx, var_name in enumerate(VARS):
                arr = ds_block[var_name].values
                if var_name == "tmean":
                    arr = arr.astype(np.float32) / 10.0
                data_dict[(year_idx, month, var_idx)] = arr
            ds.close()
            ds_block.close()
    return data_dict
