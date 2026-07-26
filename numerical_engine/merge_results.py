#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/merge_results.py
جمع‌آوری نتایج ایستگاه‌ها با پشتیبانی از موازی‌سازی
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from zarr_schema import create_empty_block_result, VAR_NAMES

def merge_station_result(block_result, station_result, local_idx):
    for name in VAR_NAMES:
        block_result[name][:, local_idx] = station_result[name]

def analyze_station_wrapper(args):
    """wrapper برای استفاده در ThreadPoolExecutor"""
    station_data, window_table, var_idx, local_idx = args
    from numerical_engine.analyze_station import analyze_station
    try:
        station_result = analyze_station(station_data, window_table, var_idx)
        return local_idx, station_result
    except Exception as e:
        return local_idx, None

def create_and_merge_results(block_data, window_table, var_idx):
    """
    پردازش همه ایستگاه‌ها با موازی‌سازی (در صورت فعال بودن)
    """
    block_size = block_data.shape[0]
    block_result = create_empty_block_result(block_size)

    # خواندن تعداد کارگرهای موازی از محیط یا استفاده از پیش‌فرض
    n_workers = int(os.environ.get("PARALLEL_WORKERS", "4"))
    use_parallel = os.environ.get("USE_PARALLEL", "1") == "1"

    if not use_parallel or block_size < 50:
        # حالت سریال (برای بلوک‌های کوچک)
        for local_idx in range(block_size):
            station_data = block_data[local_idx]
            station_result = analyze_station(station_data, window_table, var_idx)
            merge_station_result(block_result, station_result, local_idx)
        return block_result

    # حالت موازی
    args_list = [(block_data[i], window_table, var_idx, i) for i in range(block_size)]
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(analyze_station_wrapper, args) for args in args_list]
        for future in as_completed(futures):
            local_idx, station_result = future.result()
            if station_result is not None:
                merge_station_result(block_result, station_result, local_idx)
            completed += 1
            if completed % 50 == 0:
                print(f"   ⏳ پردازش {completed}/{block_size} ایستگاه...")

    return block_result
