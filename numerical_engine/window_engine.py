#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/window_engine.py
================================================================================
موتور استخراج پنجره با دو حالت: عادی (۵ روزه) و حدی (بیشینه/کمینه مطلق)
نسخه‌ی نهایی با محافظت کامل در برابر داده‌های غیر آرایه‌ای
================================================================================
ورژن: ۳.۱
"""

import numpy as np
from constants import N_YEARS, N_DAYS, MIN_VALID_VALUES


def extract_window_values_fast(station_data, window_table, var_idx):
    """
    حالت عادی: استخراج همه مقادیر پنجره ۵ روزه (۱۵۰ مقدار)
    """
    N_DAYS_LOCAL = 366
    results = []

    # ============================================================
    # محافظت: اطمینان از اینکه station_data یک آرایه است
    # ============================================================
    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)

    # اگر station_data یک عدد بود، به شکل (1, 1, 1) تبدیل کن
    if station_data.ndim == 0:
        station_data = station_data.reshape(1, 1, 1)
    elif station_data.ndim == 1:
        # اگر یک‌بعدی است، سعی کن به شکل (N_YEARS, N_DAYS, n_vars) تبدیل کن
        try:
            n_vars = station_data.shape[0] // (N_YEARS * N_DAYS_LOCAL)
            if n_vars > 0 and station_data.size == N_YEARS * N_DAYS_LOCAL * n_vars:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)
            else:
                station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
        except:
            station_data = np.full((N_YEARS, N_DAYS_LOCAL, 3), np.nan, dtype=np.float32)
    elif station_data.ndim == 2:
        # اگر دو‌بعدی است، سعی کن به شکل (N_YEARS, N_DAYS, n_vars) تبدیل کن
        if station_data.shape == (N_YEARS, N_DAYS_LOCAL):
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, 1)
        elif station_data.shape[0] == N_YEARS * N_DAYS_LOCAL:
            n_vars = station_data.shape[1] if station_data.ndim > 1 else 1
            station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)

    # اگر station_data خالی یا همه NaN بود، همه‌ی روزها None برگردان
    if station_data.size == 0 or np.all(np.isnan(station_data)):
        return [None] * N_DAYS_LOCAL

    # حلقه‌ی اصلی
    for doy_idx in range(N_DAYS_LOCAL):
        try:
            if 2 <= doy_idx <= N_DAYS_LOCAL - 3:
                window = station_data[:, doy_idx - 2 : doy_idx + 3, var_idx]
            else:
                raw_indices = window_table[doy_idx]
                safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]
                window = station_data[:, safe_indices, var_idx]

            # اطمینان از اینکه window یک آرایه است
            if not isinstance(window, np.ndarray):
                window = np.array(window)

            # اگر window یک عدد بود (اسکالر)، به آرایه تبدیل کن
            if window.ndim == 0:
                window = window.reshape(1)

            # flat کردن
            values = window.reshape(-1)

            # حذف NaN
            clean = values[~np.isnan(values)]

            if len(clean) >= MIN_VALID_VALUES:
                results.append(clean.astype(np.float64))
            else:
                results.append(None)

        except Exception:
            results.append(None)

    return results


def extract_extreme_values_fast(station_data, window_table, var_idx):
    """
    حالت حدی: برای هر سال، بیشینه و کمینه مطلق را از پنجره ۵ روزه استخراج میکند.
    خروجی: برای هر روز، دو آرایه (بیشینهها و کمینهها) به صورت مجزا.
    """
    N_DAYS_LOCAL = 366
    max_results = []
    min_results = []

    # ============================================================
    # محافظت: اطمینان از اینکه station_data یک آرایه است
    # ============================================================
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

    # حلقه‌ی اصلی
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