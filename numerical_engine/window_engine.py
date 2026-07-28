#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/window_engine.py
================================================================================
موتور استخراج پنجره با دو حالت: عادی (۵ روزه) و حدی (بیشینه/کمینه مطلق)
================================================================================
ورژن: ۳.۰
"""

import numpy as np
from constants import N_YEARS, N_DAYS, MIN_VALID_VALUES

def extract_window_values_fast(station_data, window_table, var_idx):
    N_DAYS_LOCAL = 366
    results = []
    for doy_idx in range(N_DAYS_LOCAL):
        if 2 <= doy_idx <= N_DAYS_LOCAL - 3:
            window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            values = window.reshape(-1)
        else:
            # استفاده از window_table ولی با محدود کردن اندیس‌ها
            raw_indices = window_table[doy_idx]
            # اطمینان از اینکه همه‌ی اندیس‌ها در بازه 0 تا N_DAYS_LOCAL-1 هستند
            safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]
            window = station_data[:, safe_indices, var_idx]
            values = window.reshape(-1)
        clean = values[~np.isnan(values)]
        if len(clean) >= MIN_VALID_VALUES:
            results.append(clean.astype(np.float64))
        else:
            results.append(None)
    return results
def extract_extreme_values_fast(station_data, window_table, var_idx):
    """
    حالت حدی: برای هر سال، بیشینه و کمینه مطلق را از پنجره ۵ روزه استخراج میکند.
    خروجی: برای هر روز، دو آرایه (بیشینهها و کمینهها) به صورت مجزا.
    """
    max_results = []
    min_results = []

    for doy_idx in range(N_DAYS):
        window_days = window_table[doy_idx]
        if doy_idx >= 2 and doy_idx <= N_DAYS - 3:
            window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
        else:
            window = station_data[:, window_days, var_idx]

        max_vals = np.nanmax(window, axis=1)
        min_vals = np.nanmin(window, axis=1)

        max_clean = max_vals[~np.isnan(max_vals)]
        min_clean = min_vals[~np.isnan(min_vals)]

        if len(max_clean) >= MIN_VALID_VALUES:
            max_results.append(max_clean.astype(np.float64))
        else:
            max_results.append(None)

        if len(min_clean) >= MIN_VALID_VALUES:
            min_results.append(min_clean.astype(np.float64))
        else:
            min_results.append(None)

    return max_results, min_results

# برای سازگاری با نسخه‌های قدیمی
extract_window_values = extract_window_values_fast
