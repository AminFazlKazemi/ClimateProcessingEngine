#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/window_engine.py
================================================================================
موتور استخراج پنجره مستقل.
ورودی: station_data (N_YEARS, N_DAYS, N_VARS)
خروجی: برای هر روز، ۱۵۵ مقدار (با NaN حذف شده)
================================================================================
ورژن: 2.0 - نهایی
"""

import numpy as np
from constants import N_YEARS, N_DAYS, MIN_VALID_VALUES

def extract_window_values_fast(station_data, window_table, var_idx):
    results = []
    for doy_idx in range(N_DAYS):
        window_days = window_table[doy_idx]
        if doy_idx >= 2 and doy_idx <= N_DAYS - 3:
            window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            values = window.reshape(-1)
        else:
            values = station_data[:, window_days, var_idx].reshape(-1)
        clean = values[~np.isnan(values)]
        if len(clean) >= MIN_VALID_VALUES:
            results.append(clean.astype(np.float64))
        else:
            results.append(None)
    return results
