#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/outlier_detection.py
================================================================================
آزمون گرابز (Grubbs' test) برای تشخیص و حذف اعداد پرت
نسخه اصلاح‌شده: بازگرداندن ایندکس‌های حذف‌شده برای لاگ‌گیری دقیق
================================================================================
"""

import numpy as np
from scipy.stats import t as t_dist

def grubbs_test(data, alpha=0.05, max_iter=3):
    """
    Grubbs' test for outliers (two-sided) – همراه با ایندکس‌های حذف‌شده
    
    بازگشت:
        cleaned_data: داده‌های پالایش‌شده
        removed_indices: ایندکس‌های حذف‌شده (نسبت به داده‌ی اصلی)
        removed_values: مقادیر حذف‌شده
    """
    data = np.array(data)
    original_indices = np.arange(len(data))
    cleaned = data[~np.isnan(data)]  # حذف NaN
    # اگر NaN حذف شد، ایندکس‌های اصلی را هم به‌روز کن
    if len(cleaned) < len(data):
        valid_mask = ~np.isnan(data)
        original_indices = original_indices[valid_mask]
    
    removed_indices = []
    removed_values = []
    
    iteration = 0
    while len(cleaned) > 3 and iteration < max_iter:
        n = len(cleaned)
        mean = np.mean(cleaned)
        std = np.std(cleaned, ddof=1)
        
        if std == 0:
            break
        
        abs_dev = np.abs(cleaned - mean)
        max_idx = np.argmax(abs_dev)
        max_val = cleaned[max_idx]
        g_calc = abs_dev[max_idx] / std
        
        try:
            t_crit = t_dist.ppf(1 - alpha / (2 * n), n - 2)
            g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        except:
            break
        
        if g_calc > g_crit:
            # ثبت ایندکس اصلی و مقدار حذف‌شده
            removed_indices.append(original_indices[max_idx])
            removed_values.append(max_val)
            # حذف از cleaned و به‌روزرسانی original_indices
            cleaned = np.delete(cleaned, max_idx)
            original_indices = np.delete(original_indices, max_idx)
            iteration += 1
        else:
            break
    
    return cleaned, removed_indices, removed_values

def grubbs_test_batch(data_2d, alpha=0.05, max_iter=3):
    cleaned_windows = []
    removed_summary = []
    for window in data_2d:
        if window is None or len(window) < 5:
            cleaned_windows.append(window)
            removed_summary.append([])
            continue
        cleaned, removed_idx, removed_val = grubbs_test(window, alpha, max_iter)
        cleaned_windows.append(cleaned)
        removed_summary.append(list(zip(removed_idx, removed_val)))
    return cleaned_windows, removed_summary