#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/merge_results.py
================================================================================
جمع‌آوری نتایج ایستگاه‌ها در block_result.
================================================================================
"""

from zarr_schema import create_empty_block_result, VAR_NAMES

def merge_station_result(block_result, station_result, local_idx):
    """ادغام نتیجه یک ایستگاه در block_result"""
    for name in VAR_NAMES:
        block_result[name][:, local_idx] = station_result[name]

def create_and_merge_results(block_data, window_table, var_idx):
    """
    پردازش کامل یک بلوک: تحلیل همه ایستگاه‌ها و جمع‌آوری نتایج.

    پارامترها:
        block_data: ndarray shape=(block_size, N_YEARS, N_DAYS, N_VARS)
        window_table: لیست پنجره‌ها
        var_idx: اندیس متغیر برای برازش

    خروجی: dict {name: ndarray(shape=(N_DAYS, block_size))}
    """
    from numerical_engine.analyze_station import analyze_station

    block_size = block_data.shape[0]
    block_result = create_empty_block_result(block_size)

    for local_idx in range(block_size):
        station_data = block_data[local_idx]
        station_result = analyze_station(station_data, window_table, var_idx)
        merge_station_result(block_result, station_result, local_idx)

    return block_result

# ============================================================================
# توابع سازگاری با کدهای قدیمی
# ============================================================================

def merge_results(block_data, window_table, var_idx):
    """
    Wrapper برای create_and_merge_results (سازگاری با نسخه‌های قدیمی)
    """
    return create_and_merge_results(block_data, window_table, var_idx)

if __name__ == "__main__":
    print("✅ merge_results.py loaded successfully.")
