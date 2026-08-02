#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/window_engine.py
================================================================================
موتور استخراج پنجره با دو حالت: عادی (۵ روزه) و حدی (بیشینه/کمینه مطلق)
نسخه ۴.۲ – با پشتیبانی از داده‌های خام (برای ثبت outlier)
================================================================================
"""

import numpy as np
from constants import N_YEARS, N_DAYS, MIN_VALID_VALUES

# ============================================================
# محدوده‌ی دمای معقول (داده‌های ورودی ×۱۰ هستند)
# ============================================================
MIN_TEMP_RAW = -400   # معادل -40 درجه سانتی‌گراد
MAX_TEMP_RAW = 550    # معادل 55 درجه سانتی‌گراد


def extract_window_values_fast(station_data, window_table, var_idx):
    """
    حالت عادی: استخراج همه مقادیر پنجره ۵ روزه (۱۵۰ مقدار)
    با فیلتر دمای غیرمنطقی (بر اساس داده‌های خام ×۱۰)
    """
    N_DAYS_LOCAL = 366
    results = []

    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)
    if station_data.ndim == 0:
        station_data = station_data.reshape(1, 1, 1)
    elif station_data.ndim == 1:
        try:
            n_vars = station_data.shape[0] // (N_YEARS * N_DAYS_LOCAL)
            if n_vars > 0 and station_data.size == N_YEARS * N_DAYS_LOCAL * n_vars:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)
            else:
                station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
        except:
            station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
    elif station_data.ndim == 2:
        if station_data.shape == (N_YEARS, N_DAYS_LOCAL):
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, 1)
        elif station_data.shape[0] == N_YEARS * N_DAYS_LOCAL:
            n_vars = station_data.shape[1] if station_data.ndim > 1 else 1
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)

    if station_data.size == 0 or np.all(np.isnan(station_data)):
        return [None] * N_DAYS_LOCAL

    for doy_idx in range(N_DAYS_LOCAL):
        try:
            if 2 <= doy_idx <= N_DAYS_LOCAL - 3:
                window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            else:
                raw_indices = window_table[doy_idx]
                safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]
                window = station_data[:, safe_indices, var_idx]

            if not isinstance(window, np.ndarray):
                window = np.array(window)
            if window.ndim == 0:
                window = window.reshape(1)

            values = window.reshape(-1)
            clean = values[~np.isnan(values)]
            clean = clean[(clean >= MIN_TEMP_RAW) & (clean <= MAX_TEMP_RAW)]

            if len(clean) >= MIN_VALID_VALUES:
                results.append(clean.astype(np.float64))
            else:
                results.append(None)
        except Exception:
            results.append(None)

    return results


def extract_window_values_raw(station_data, window_table, var_idx):
    """
    استخراج داده‌های خام پنجره (بدون فیلتر) برای ثبت outlierها
    خروجی: لیستی به طول ۳۶۶، هر عنصر یک آرایه از تمام مقادیر (شامل NaN و outlier)
    """
    N_DAYS_LOCAL = 366
    results = []

    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)
    if station_data.ndim == 0:
        station_data = station_data.reshape(1, 1, 1)
    elif station_data.ndim == 1:
        try:
            n_vars = station_data.shape[0] // (N_YEARS * N_DAYS_LOCAL)
            if n_vars > 0 and station_data.size == N_YEARS * N_DAYS_LOCAL * n_vars:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)
            else:
                station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
        except:
            station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
    elif station_data.ndim == 2:
        if station_data.shape == (N_YEARS, N_DAYS_LOCAL):
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, 1)
        elif station_data.shape[0] == N_YEARS * N_DAYS_LOCAL:
            n_vars = station_data.shape[1] if station_data.ndim > 1 else 1
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)

    if station_data.size == 0 or np.all(np.isnan(station_data)):
        return [None] * N_DAYS_LOCAL

    for doy_idx in range(N_DAYS_LOCAL):
        try:
            if 2 <= doy_idx <= N_DAYS_LOCAL - 3:
                window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            else:
                raw_indices = window_table[doy_idx]
                safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]
                window = station_data[:, safe_indices, var_idx]

            if not isinstance(window, np.ndarray):
                window = np.array(window)
            if window.ndim == 0:
                window = window.reshape(1)

            values = window.reshape(-1)
            results.append(values.astype(np.float64))
        except Exception:
            results.append(None)

    return results


def extract_extreme_values_fast(station_data, window_table, var_idx):
    """حالت حدی (بدون تغییر)"""
    N_DAYS_LOCAL = 366
    max_results = []
    min_results = []

    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)
    if station_data.ndim == 0:
        station_data = station_data.reshape(1, 1, 1)
    elif station_data.ndim == 1:
        try:
            n_vars = station_data.shape[0] // (N_YEARS * N_DAYS_LOCAL)
            if n_vars > 0 and station_data.size == N_YEARS * N_DAYS_LOCAL * n_vars:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)
            else:
                station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
        except:
            station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
    elif station_data.ndim == 2:
        if station_data.shape == (N_YEARS, N_DAYS_LOCAL):
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, 1)
        elif station_data.shape[0] == N_YEARS * N_DAYS_LOCAL:
            n_vars = station_data.shape[1] if station_data.ndim > 1 else 1
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)

    if station_data.size == 0 or np.all(np.isnan(station_data)):
        return [None] * N_DAYS_LOCAL, [None] * N_DAYS_LOCAL

    for doy_idx in range(N_DAYS_LOCAL):
        try:
            if 2 <= doy_idx <= N_DAYS_LOCAL - 3:
                window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            else:
                raw_indices = window_table[doy_idx]
                safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]
                window = station_data[:, safe_indices, var_idx]

            if not isinstance(window, np.ndarray):
                window = np.array(window)

            max_vals = np.nanmax(window, axis=1)
            min_vals = np.nanmin(window, axis=1)

            max_clean = max_vals[~np.isnan(max_vals)]
            min_clean = min_vals[~np.isnan(min_vals)]

            max_clean = max_clean[(max_clean >= MIN_TEMP_RAW) & (max_clean <= MAX_TEMP_RAW)]
            min_clean = min_clean[(min_clean >= MIN_TEMP_RAW) & (min_clean <= MAX_TEMP_RAW)]

            if len(max_clean) >= MIN_VALID_VALUES:
                max_results.append(max_clean.astype(np.float64))
            else:
                max_results.append(None)

            if len(min_clean) >= MIN_VALID_VALUES:
                min_results.append(min_clean.astype(np.float64))
            else:
                min_results.append(None)

        except Exception:
            max_results.append(None)
            min_results.append(None)

    return max_results, min_results


# برای سازگاری با نسخه‌های قدیمی
extract_window_values = extract_window_values_fast