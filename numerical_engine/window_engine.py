#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/window_engine.py
================================================================================
موتور استخراج پنجره با دو حالت: عادی (۵ روزه) و حدی (بیشینه/کمینه مطلق)
نسخه ۶.۱ - با لاگ‌گیری دقیق سال و روز برای آزمون گرابز (اصلاح شده بر روی فایل اصلی ۲۱۸ خطی)
================================================================================
"""

import numpy as np
from constants import N_YEARS, N_DAYS, MIN_VALID_VALUES, VARS  # 🔴 اضافه شدن VARS

# ============================================================
# ایمپورت آزمون گرابز و لاگر
# ============================================================
try:
    from numerical_engine.outlier_detection import grubbs_test
    from monitoring.outlier_logger import log_outlier  # 🔴 اضافه شدن لاگر
    GRUBBS_AVAILABLE = True
except ImportError:
    GRUBBS_AVAILABLE = False
    print("⚠️ outlier_detection.py یا outlier_logger.py یافت نشد. آزمون گرابز/لاگ غیرفعال است.")


def extract_window_values_fast(station_data, year_list, window_table, var_idx, station_idx):  # 🔴 اضافه شدن آرگومان‌های year_list و station_idx
    """
    حالت عادی: استخراج همه مقادیر پنجره ۵ روزه (۱۵۰ مقدار)
    با اعمال آزمون گرابز برای حذف اعداد پرت و لاگ‌گیری دقیق سال/روز.
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
                safe_indices = [doy_idx - 2, doy_idx - 1, doy_idx, doy_idx + 1, doy_idx + 2]  # 🔴 تعریف safe_indices برای روزهای وسط
            else:
                raw_indices = window_table[doy_idx]
                safe_indices = [i % N_DAYS_LOCAL for i in raw_indices]  # 🔴 تعریف safe_indices برای روزهای ابتدایی/انتهایی
                window = station_data[:, safe_indices, var_idx]

            if not isinstance(window, np.ndarray):
                window = np.array(window)
            if window.ndim == 0:
                window = window.reshape(1)

            values = window.reshape(-1)
            clean = values[~np.isnan(values)]

            # ============================================================
            # اعمال آزمون گرابز و لاگ‌گیری دقیق
            # ============================================================
            if GRUBBS_AVAILABLE and len(clean) >= 5:
                # دریافت ایندکس‌های حذف‌شده از تابع اصلاح‌شده گرابز
                clean, removed_indices, removed_values = grubbs_test(clean, alpha=0.05, max_iter=3)
                
                # 🔴 محاسبه سال و روز برای هر پرت حذف‌شده و ثبت در لاگر
                for rem_idx, rem_val in zip(removed_indices, removed_values):
                    # هر 5 عنصر مربوط به یک سال است (چون پنجره 5 روزه است)
                    idx_year = rem_idx // 5
                    idx_in_window = rem_idx % 5
                    
                    actual_year = year_list[idx_year]
                    actual_day = safe_indices[idx_in_window] + 1  # 1-based day
                    
                    log_outlier(
                        station_idx=station_idx,
                        year=actual_year,
                        day_idx=actual_day,
                        value=rem_val,
                        var_name=VARS[var_idx]
                    )
            elif GRUBBS_AVAILABLE and len(clean) < 5:
                pass  # داده‌های خیلی کم → بدون آزمون گرابز

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
    خروجی: لیستی به طول ۳۶۶، هر عنصر یک آرایه از تمام مقادیر (شامل NaN)
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
    """حالت حدی: استخراج بیشینه و کمینه مطلق هر سال از پنجره - بدون فیلتر"""
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

            # اعمال آزمون گرابز روی مقادیر بیشینه و کمینه
            if GRUBBS_AVAILABLE and len(max_clean) >= 5:
                max_clean, _ = grubbs_test(max_clean, alpha=0.05, max_iter=3)
            if GRUBBS_AVAILABLE and len(min_clean) >= 5:
                min_clean, _ = grubbs_test(min_clean, alpha=0.05, max_iter=3)

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


# برای سازگاری با نسخه‌های قدیمی - داده‌های خام را برمی‌گرداند
extract_window_values = extract_window_values_raw