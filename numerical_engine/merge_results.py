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
    فرض می‌کند block_result از قبل با تمام کلیدها و ابعاد صحیح مقداردهی شده است.
    اگر کلیدی وجود نداشت، هشدار داده می‌شود (اما نباید رخ دهد).
    """
    local_idx = station_idx - block_start

    for key, value in station_result.items():
        if key not in block_result:
            # اگر کلید در block_result وجود ندارد (نباید رخ دهد)، آن را با ابعاد صحیح ایجاد کن
            # اما block_size را نمی‌دانیم، بنابراین فقط هشدار می‌دهیم و از آن صرف‌نظر می‌کنیم
            print(f"⚠️ Warning: key '{key}' not found in block_result. Skipping.")
            continue
        # قرار دادن داده‌ها در ستون مربوطه
        block_result[key][:, local_idx] = value