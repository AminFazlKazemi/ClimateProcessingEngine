# -*- coding: utf-8 -*-
"""
process_block.py - پردازش یک بلوک از ایستگاه‌ها (نسخه‌ی سریال)
"""

import time
import numpy as np
import gc
from tqdm import tqdm
from monitoring.logger import logger
from monitoring.checkpoint import save_checkpoint
from io_pipeline.read_month_files import read_month_files
from io_pipeline.assemble_block import assemble_block
from io_pipeline.validate_block import validate_block
from numerical_engine.analyze_station import analyze_station
from numerical_engine.merge_results import merge_station_result
from result_pipeline.validate_result import validate_result
from result_pipeline.write_block import write_block_safe
from constants import VARS, VAR_INDEX_FOR_FIT, N_DAYS
from zarr_schema import create_empty_block_result


def process_block(block_start, block_end, block_idx, file_map, doy_table, window_table,
                  year_list, root, var_idx, last_checkpoint_station=0, adapter=None):
    """
    پردازش یک بلوک از ایستگاه‌ها (به صورت سریال).
    """
    block_size = block_end - block_start
    logger.info(f"📦 Block {block_idx}: stations {block_start} - {block_end} ({block_size} stations)")

    times = {"load": 0, "analyze": 0, "write": 0}
    t0 = time.time()

    # ============================================================
    # ۱. خواندن داده‌ها (با یا بدون Adapter)
    # ============================================================
    logger.info("   📂 Loading...")

    if adapter is not None:
        logger.info("   📂 Using Data Adapter for loading...")
        data_dict = {}

        if hasattr(adapter, 'load_block_all_vars'):
            logger.info("   ⚡ Using fast multi-variable loading...")
            total_files = sum(1 for year in year_list for month in range(1, 13) if (year, month) in file_map)
            pbar = tqdm(total=total_files, desc="   Loading Zarr files", unit="file", position=0, leave=True)

            for year in year_list:
                for month in range(1, 13):
                    key = (year, month)
                    if key not in file_map:
                        continue
                    combined_data = adapter.load_block_all_vars(
                        block_start=block_start,
                        block_size=block_size,
                        year_idx=year_list.index(year) if year in year_list else 0,
                        month=month
                    )
                    if combined_data is not None:
                        data_dict[key] = combined_data
                    pbar.update(1)
            pbar.close()
        else:
            logger.info("   🐢 Using fallback single-variable loading...")
            n_vars = len(VARS)
            total_files = sum(1 for year in year_list for month in range(1, 13) if (year, month) in file_map)
            pbar = tqdm(total=total_files, desc="   Loading Zarr files (single-var)", unit="file", position=0, leave=True)

            for year in year_list:
                for month in range(1, 13):
                    key = (year, month)
                    if key not in file_map:
                        continue
                    combined_data = None
                    for v in range(n_vars):
                        var_data = adapter.load_block(
                            block_start=block_start,
                            block_size=block_size,
                            year_idx=year_list.index(year) if year in year_list else 0,
                            month=month,
                            var_idx=v
                        )
                        if var_data is not None:
                            if var_data.ndim == 2 and var_data.shape[0] == block_size:
                                var_data = var_data.T
                            elif var_data.ndim == 1:
                                var_data = var_data.reshape(-1, 1)
                            if combined_data is None:
                                days = var_data.shape[0]
                                combined_data = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
                            combined_data[:, :, v] = var_data
                    if combined_data is not None:
                        data_dict[key] = combined_data
                    pbar.update(1)
            pbar.close()

        if not data_dict:
            logger.warning(f"   ⚠️ No data loaded via adapter for block {block_idx}")
            return None
    else:
        data_dict = read_month_files(block_start, block_size, file_map, year_list)
        if not data_dict:
            logger.warning(f"   ⚠️ No data loaded for block {block_idx}")
            return None

    times["load"] = time.time() - t0

    # ============================================================
    # ۲. مونتاژ داده‌ها
    # ============================================================
    block_data = assemble_block(data_dict, doy_table, block_size, year_list, var_idx)
    if hasattr(block_data, 'values'):
        block_data = block_data.values
    elif not isinstance(block_data, np.ndarray):
        block_data = np.array(block_data)

    if block_data.size > 0:
        validate_block(block_data, block_start, block_size, f"Block {block_idx}")

    # ============================================================
    # ۳. تحلیل سریال هر ایستگاه
    # ============================================================
    logger.info("   ⚙️ Analyzing (serial)...")

    block_result = create_empty_block_result(block_size)
    N_YEARS = len(year_list)
    N_DAYS_LOCAL = 366
    n_vars = len(VARS)

    if block_data.ndim != 4:
        logger.error(f"   ❌ Invalid block_data shape: {block_data.shape}")
        raise ValueError(f"Invalid block_data shape: {block_data.shape}")

    t_analyze_start = time.time()
    pbar_stations = tqdm(total=block_size, desc="   Analyzing stations", unit="station", position=0, leave=True)

    for local_idx in range(block_size):
        station_idx = block_start + local_idx

        if block_idx == 0 and last_checkpoint_station is not None and last_checkpoint_station > 0 and station_idx < last_checkpoint_station:
            pbar_stations.update(1)
            continue

        station_data = block_data[local_idx, :, :, :]

        if not isinstance(station_data, np.ndarray):
            station_data = np.array(station_data)

        if station_data.ndim == 0:
            station_data = station_data.reshape(1, 1, 1)
        elif station_data.ndim == 1:
            try:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)
            except ValueError:
                logger.warning(f"   ⚠️ Station {station_idx}: cannot reshape data")
                station_data = np.full((N_YEARS, N_DAYS_LOCAL, n_vars), np.nan, dtype=np.float32)
        elif station_data.ndim == 2:
            if station_data.shape[0] == N_YEARS * N_DAYS_LOCAL and station_data.shape[1] == n_vars:
                station_data = station_data.reshape(N_YEARS, N_DAYS_LOCAL, n_vars)

        try:
            # 🔴 اصلاح مهم: پاس دادن آرگومان پنجم (station_idx) به analyze_station
            result = analyze_station(station_data, year_list, window_table, var_idx, station_idx)
            if result is not None:
                merge_station_result(block_result, result, local_idx)
        except Exception as e:
            logger.warning(f"   ⚠️ Station {station_idx} failed: {e}")

        if (station_idx - block_start + 1) % 100 == 0 or station_idx == block_end - 1:
            save_checkpoint(block_idx, station_idx + 1)

        pbar_stations.update(1)

    pbar_stations.close()
    times["analyze"] = time.time() - t_analyze_start

    # ============================================================
    # ۴. اعتبارسنجی و نوشتن
    # ============================================================
    if block_result:
        validate_result(block_result, block_start, block_size)

    t0 = time.time()
    try:
        write_block_safe(root, block_result, block_start, block_end, validate=False, async_mode=False)
        times["write"] = time.time() - t0
    except Exception as e:
        logger.error(f"   ❌ Write failed: {e}")
        save_checkpoint(block_idx, block_start)
        raise IOError(f"Write failed: {e}")

    save_checkpoint(block_idx, block_end - 1)

    total_time = sum(times.values())
    logger.info(f"   ✅ Block {block_idx} completed in {total_time:.1f}s")
    logger.info(f"       Load: {times['load']:.1f}s | Analyze: {times['analyze']:.1f}s | Write: {times['write']:.1f}s")
    if total_time > 0:
        logger.info(f"       Stations/sec: {block_size / total_time:.1f}")

    del data_dict, block_data, block_result
    gc.collect()
    return True