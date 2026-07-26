#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
orchestrator/process_block.py
================================================================================
Orchestrator اصلی.
فقط هماهنگ‌کننده است. محاسبه‌ای انجام نمی‌دهد.
================================================================================
"""

import time
import gc
from io_pipeline.read_month_files import read_month_files
from io_pipeline.assemble_block import assemble_block
from io_pipeline.validate_block import validate_block, print_validation_report
from numerical_engine.merge_results import create_and_merge_results
from result_pipeline.validate_result import validate_result, print_validation_report as print_result_report
from result_pipeline.write_block import write_block_safe
from monitoring.checkpoint import save_checkpoint
from monitoring.logger import logger
from constants import VALIDATE_AFTER_LOAD, VALIDATE_BEFORE_WRITE, VALIDATE_EVERY_N_BLOCKS

class FitError(Exception):
    pass

class DataError(Exception):
    pass

class IOError(Exception):
    pass

def process_block(block_start, block_end, block_idx, file_map, doy_table, window_table, year_list, root, var_idx, last_checkpoint_station=None):
    block_size = block_end - block_start
    start_time = time.time()
    times = {}

    logger.info(f"📦 Block {block_idx}: stations {block_start:,} - {block_end:,} ({block_size:,} stations)")

    try:
        t0 = time.time()
        logger.info("   📂 Loading...")
        data_dict = read_month_files(block_start, block_size, file_map, year_list)
        block_data = assemble_block(data_dict, doy_table, block_size, year_list)
        times["load"] = time.time() - t0
    except Exception as e:
        logger.error(f"   ❌ Load failed: {e}")
        save_checkpoint(block_idx, block_start)
        raise IOError(f"Load failed: {e}")

    if VALIDATE_AFTER_LOAD or block_idx % VALIDATE_EVERY_N_BLOCKS == 0:
        t0 = time.time()
        logger.info("   🔍 Validating block...")
        try:
            report = validate_block(block_data, block_start, block_size, strict=True)
            print_validation_report(report)
            times["validate_load"] = time.time() - t0
        except Exception as e:
            logger.error(f"   ❌ Validation failed: {e}")
            save_checkpoint(block_idx, block_start)
            raise DataError(f"Block validation failed: {e}")

    t0 = time.time()
    logger.info("   ⚙️ Analyzing...")
    try:
        block_result = create_and_merge_results(block_data, window_table, var_idx)
        times["analyze"] = time.time() - t0
    except Exception as e:
        logger.error(f"   ❌ Analysis failed: {e}")
        save_checkpoint(block_idx, block_start)
        raise FitError(str(e))

    if VALIDATE_BEFORE_WRITE or block_idx % VALIDATE_EVERY_N_BLOCKS == 0:
        t0 = time.time()
        logger.info("   🔍 Validating results...")
        try:
            report = validate_result(block_result, block_start, block_size, strict=True)
            print_result_report(report)
            times["validate_write"] = time.time() - t0
        except Exception as e:
            logger.error(f"   ❌ Result validation failed: {e}")
            save_checkpoint(block_idx, block_start)
            raise DataError(f"Result validation failed: {e}")

    t0 = time.time()
    logger.info("   💾 Writing to Zarr...")
    try:
        write_block_safe(root, block_result, block_start, block_end, validate=False)
        times["write"] = time.time() - t0
    except Exception as e:
        logger.error(f"   ❌ Write failed: {e}")
        save_checkpoint(block_idx, block_start)
        raise IOError(f"Write failed: {e}")

    del data_dict, block_data, block_result
    gc.collect()

    save_checkpoint(block_idx, block_end - 1)

    total_time = time.time() - start_time
    stations_per_sec = block_size / times["analyze"] if times["analyze"] > 0 else 0

    logger.info(f"   ✅ Block {block_idx} completed in {total_time:.1f}s")
    logger.info(f"      Load: {times['load']:.1f}s | Analyze: {times['analyze']:.1f}s | Write: {times['write']:.1f}s")
    logger.info(f"      Stations/sec: {stations_per_sec:.1f}")


    # ====================================================================
    # ذخیره داده‌های پنجره‌ای در Zarr میانی (فقط یک بار)
    # ====================================================================
    if not os.environ.get("SKIP_WINDOW_CACHE", "0") == "1":
        try:
            from numerical_engine.window_engine import extract_window_values_fast
            cache_path = os.path.join(os.path.dirname(root.store.path), "window_cache.zarr")
            import zarr
            cache_root = zarr.open(cache_path, mode="a")
            # تعیین تعداد کل ایستگاه‌ها (از root بگیریم)
            n_stations_total = root["best_dist"].shape[1]
            if "window_data" not in cache_root:
                cache_root.create_array(
                    "window_data",
                    shape=(n_stations_total, N_DAYS, MAX_VALUES_PER_FIT),
                    chunks=(100, 366, 155),
                    dtype=np.float32,
                    fill_value=np.nan,
                )
            for local_idx in range(block_size):
                station_data = block_data[local_idx]
                windows = extract_window_values_fast(station_data, window_table, var_idx)
                global_idx = block_start + local_idx
                for doy_idx, vals in enumerate(windows):
                    if vals is not None:
                        vals_trim = vals[:MAX_VALUES_PER_FIT]
                        cache_root["window_data"][global_idx, doy_idx, :len(vals_trim)] = vals_trim
            logger.info("   💾 Window data cached.")
        except Exception as e:
            logger.warning(f"   ⚠️ Window caching failed: {e}")

    return {"block_idx": block_idx, "times": times, "total_time": total_time, "stations_per_sec": stations_per_sec}
