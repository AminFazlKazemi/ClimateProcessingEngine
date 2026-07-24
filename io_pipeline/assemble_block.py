#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_pipeline/assemble_block.py
================================================================================
ساخت آرایه (block_size, N_YEARS, N_DAYS, N_VARS) از داده‌های خوانده‌شده.
داده‌های نامعتبر (doy == -1) به صورت NaN باقی می‌مانند.
================================================================================
ورژن: 2.0 - نهایی
"""

import numpy as np
from constants import N_YEARS, N_DAYS, N_VARS, FLOAT_DTYPE

def assemble_block(data_dict, doy_table, block_size, year_list):
    data = np.full((block_size, N_YEARS, N_DAYS, N_VARS), np.nan, dtype=FLOAT_DTYPE)
    for year_idx in range(N_YEARS):
        for month in range(1, 13):
            for var_idx in range(N_VARS):
                key = (year_idx, month, var_idx)
                if key not in data_dict:
                    continue
                arr = data_dict[key]
                n_days = arr.shape[0]
                for day_idx in range(n_days):
                    doy = doy_table[year_idx, month, day_idx + 1]
                    if doy >= 0 and doy < N_DAYS:
                        data[:, year_idx, doy, var_idx] = arr[day_idx, :]
    return data
