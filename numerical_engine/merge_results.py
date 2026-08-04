# -*- coding: utf-8 -*-
"""
numerical_engine/merge_results.py
================================================================================
ادغام نتایج ایستگاه‌ها در یک بلوک
با مدیریت کلیدهای گم‌شده و استفاده از مقدار پیش‌فرض np.nan
نسخه‌ی اصلاح‌شده با پشتیبانی از year_list
================================================================================
"""

import numpy as np
from tqdm import tqdm
from zarr_schema import create_empty_block_result, VAR_NAMES


def merge_station_result(block_result, station_result, local_idx):
    """
    ادغام نتایج یک ایستگاه در block_result.
    اگر کلیدی در station_result وجود نداشت، مقدار آن از قبل np.nan است (تغییری نمی‌کند).
    """
    for key in VAR_NAMES:
        if key in station_result:
            value = station_result[key]
            # اگر value اسکالر است، آن را به شکل (n_days,) درآور
            if np.isscalar(value) or value.ndim == 0:
                value = np.full(366, value, dtype=np.float32)
            # اطمینان از اینکه value یک آرایه با طول مناسب است (معمولاً ۳۶۶)
            if len(value) != block_result[key].shape[0]:
                if len(value) < block_result[key].shape[0]:
                    new_val = np.full(block_result[key].shape[0], np.nan, dtype=np.float32)
                    new_val[:len(value)] = value
                    value = new_val
                else:
                    value = value[:block_result[key].shape[0]]
            block_result[key][:, local_idx] = value
        # اگر کلید موجود نبود، هیچ کاری نکن (همان NaN باقی می‌ماند)


def analyze_station_wrapper(args):
    """Wrapper برای استفاده در ThreadPoolExecutor"""
    station_data, year_list, window_table, var_idx, local_idx = args
    from numerical_engine.analyze_station import analyze_station
    try:
        # ✅ ارسال year_list به analyze_station
        station_result = analyze_station(station_data, year_list, window_table, var_idx)
        return local_idx, station_result
    except Exception as e:
        # در صورت خطا، یک دیکشنری خالی (همه NaN) برمی‌گردانیم
        from constants import VARS, N_DAYS
        var_name = VARS[var_idx] if var_idx < len(VARS) else 'tmean'
        empty = {}
        for key in VAR_NAMES:
            if key.startswith(var_name):
                if key.endswith('_count') or key.endswith('_best_dist'):
                    empty[key] = np.full(N_DAYS, -1, dtype=np.int32)
                else:
                    empty[key] = np.full(N_DAYS, np.nan, dtype=np.float32)
        return local_idx, empty


def create_and_merge_results(block_data, year_list, window_table, var_idx, use_parallel=True, n_workers=4):
    """
    پردازش همه ایستگاه‌های یک بلوک و ادغام نتایج در block_result.
    با نمایش پیشرفت و مدیریت حافظه.
    """
    block_size = block_data.shape[0]
    block_result = create_empty_block_result(block_size)

    if not use_parallel or block_size < 50:
        # حالت سریال با progress bar
        pbar = tqdm(total=block_size, desc="   Analyzing stations (serial)", unit="station")
        for local_idx in range(block_size):
            station_data = block_data[local_idx]
            # ✅ ارسال year_list به analyze_station
            station_result = analyze_station(station_data, year_list, window_table, var_idx)
            merge_station_result(block_result, station_result, local_idx)
            pbar.update(1)
            del station_data, station_result
        pbar.close()
        return block_result

    # حالت موازی با progress bar
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import gc

    args_list = [(block_data[i], year_list, window_table, var_idx, i) for i in range(block_size)]
    
    pbar = tqdm(total=block_size, desc="   Analyzing stations (parallel)", unit="station")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(analyze_station_wrapper, args): args[4] for args in args_list}
        
        for future in as_completed(futures):
            local_idx, station_result = future.result()
            if station_result is not None:
                merge_station_result(block_result, station_result, local_idx)
            pbar.update(1)
            del station_result
            gc.collect()

    pbar.close()
    return block_result