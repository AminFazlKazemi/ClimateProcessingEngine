# -*- coding: utf-8 -*-
"""
merge_results.py - ادغام نتایج تحلیل ایستگاه‌ها
"""

import numpy as np


def merge_results(block_result, station_result, station_idx, block_start):
    """
    ادغام نتایج یک ایستگاه در نتایج بلوک

    Parameters
    ----------
    block_result : dict
        دیکشنری نتایج بلوک (کلید: نام متغیر، مقدار: آرایه (N_DAYS, block_size))
    station_result : dict
        نتایج ایستگاه (کلید: نام متغیر، مقدار: آرایه (N_DAYS,))
    station_idx : int
        ایندکس جهانی ایستگاه
    block_start : int
        ایندکس شروع بلوک
    """
    local_idx = station_idx - block_start

    for key, value in station_result.items():
        if key not in block_result:
            # مقداردهی اولیه با NaN
            N_DAYS = len(value)
            block_result[key] = np.full((N_DAYS, block_start + 1), np.nan, dtype=np.float32)
        # اگر ابعاد ناهماهنگ است، توسعه دهید
        if block_result[key].shape[1] <= local_idx:
            # افزایش بعد دوم
            new_shape = (block_result[key].shape[0], local_idx + 1)
            new_array = np.full(new_shape, np.nan, dtype=np.float32)
            new_array[:, :block_result[key].shape[1]] = block_result[key]
            block_result[key] = new_array

        # قرار دادن داده‌ها در ستون مربوطه
        block_result[key][:, local_idx] = value
