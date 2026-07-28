#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/merge_results.py
================================================================================
ادغام نتایج یک ایستگاه در نتایج بلوک
================================================================================
"""

import numpy as np

def merge_results(block_result, station_result, station_idx, block_start):
    """
    ادغام نتایج یک ایستگاه در نتایج بلوک (نسخه قدیمی - برای سازگاری)
    """
    merge_station_result(block_result, station_result, station_idx, block_start)

def merge_station_result(block_result, station_result, station_idx, block_start):
    """
    ادغام نتایج یک ایستگاه در نتایج بلوک
    اگر کلیدی در station_result وجود نداشت، از آن صرف‌نظر می‌کند.
    """
    local_idx = station_idx - block_start

    for key, value in station_result.items():
        if key not in block_result:
            # اگر کلید در block_result وجود ندارد، آن را با ابعاد مناسب ایجاد کن
            N_DAYS = len(value)
            block_result[key] = np.full((N_DAYS, block_start + 1), np.nan, dtype=np.float32)
        # اگر ابعاد ناهماهنگ است، توسعه دهید
        if block_result[key].shape[1] <= local_idx:
            new_shape = (block_result[key].shape[0], local_idx + 1)
            new_array = np.full(new_shape, np.nan, dtype=np.float32)
            new_array[:, :block_result[key].shape[1]] = block_result[key]
            block_result[key] = new_array

        # قرار دادن داده‌ها در ستون مربوطه
        block_result[key][:, local_idx] = value