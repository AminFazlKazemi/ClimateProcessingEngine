# -*- coding: utf-8 -*-
"""
window_engine.py - محاسبه پنجره‌های روزانه
"""

import numpy as np


def compute_windows(var_data, window_table, year_list):
    """
    محاسبه پنجره‌های روزانه برای داده‌های یک متغیر

    Parameters
    ----------
    var_data : np.ndarray
        داده‌های متغیر با shape (N_YEARS, N_DAYS)
    window_table : dict or np.ndarray
        جدول پنجره‌ها (شامل ایندکس روزهای هر پنجره)
    year_list : list
        لیست سال‌ها

    Returns
    -------
    windows : list of np.ndarray
        لیستی از آرایه‌ها برای هر روز (هر آرایه شامل داده‌های پنجره است)
    """
    N_YEARS, N_DAYS = var_data.shape
    windows = []

    # اگر window_table یک دیکشنری است که ایندکس روزهای هر پنجره را دارد
    if isinstance(window_table, dict):
        for day_idx in range(N_DAYS):
            window_indices = window_table.get(day_idx, [day_idx])
            window_data = []
            for yr in range(N_YEARS):
                for d in window_indices:
                    if 0 <= d < N_DAYS:
                        val = var_data[yr, d]
                        if not np.isnan(val):
                            window_data.append(val)
            windows.append(np.array(window_data))
    else:
        # اگر window_table یک آرایه است که روزهای پنجره را مشخص می‌کند
        # فرض می‌کنیم window_table.shape = (N_DAYS, window_size)
        for day_idx in range(N_DAYS):
            if hasattr(window_table, '__getitem__'):
                window_indices = window_table[day_idx]
                if np.isscalar(window_indices):
                    window_indices = [window_indices]
            else:
                window_indices = [day_idx]
            window_data = []
            for yr in range(N_YEARS):
                for d in window_indices:
                    if 0 <= d < N_DAYS:
                        val = var_data[yr, d]
                        if not np.isnan(val):
                            window_data.append(val)
            windows.append(np.array(window_data))

    return windows
