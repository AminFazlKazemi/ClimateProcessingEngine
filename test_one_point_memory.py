#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست یک نقطه با فروشگاه سریع (zarr_format=2)
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import zarr
from zarr.storage import MemoryStore

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    YEAR_START, YEAR_END, VARS, VAR_INDEX_FOR_FIT,
    ZARR_BASE, OUTPUT_DIR, WINDOW_DAYS
)
from runtime_tables import build_runtime_tables
from orchestrator.process_block import process_block
from monitoring.logger import logger
from io_pipeline.read_month_files import clear_ds_cache
from result_pipeline.write_block import VAR_NAMES


def get_station_info(zarr_base):
    import glob
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"هیچ فایل Zarr در {zarr_base} یافت نشد")
    ds = xr.open_zarr(zarr_files[0])
    n_stations = ds.sizes["point"]
    station_ids = ds["stationid"].values if "stationid" in ds else np.arange(n_stations)
    lons = ds["lon"].values if "lon" in ds else np.full(n_stations, np.nan)
    lats = ds["lat"].values if "lat" in ds else np.full(n_stations, np.nan)
    elevs = ds["elev"].values if "elev" in ds else np.full(n_stations, np.nan)
    ds.close()
    return n_stations, station_ids, lons, lats, elevs


def build_doy_table_from_calendar_file(calendar_path, year_list):
    try:
        data = np.loadtxt(calendar_path, skiprows=1, dtype=int)
        shamsi_dates = data[:, 0]
        julian_days = data[:, 3]
        shamsi_years = shamsi_dates // 10000

        year_to_days = {}
        for year, day in zip(shamsi_years, julian_days):
            year_to_days.setdefault(year, []).append(day)

        doy_table = []
        for yr in year_list:
            if yr in year_to_days:
                days = np.array(year_to_days[yr])
                days_sorted = np.sort(days)
                if len(days_sorted) < 366:
                    logger.warning(f"سال {yr} فقط {len(days_sorted)} روز دارد. روز 366 با 366 پر می‌شود.")
                    padded = np.full(366, 366, dtype=np.float32)
                    padded[:len(days_sorted)] = days_sorted
                    doy_table.append(padded)
                else:
                    doy_table.append(days_sorted[:366])
            else:
                logger.warning(f"سال {yr} در calendar.txt یافت نشد. از doy_table ساختگی استفاده می‌شود.")
                doy_table.append(np.arange(1, 367, dtype=np.float32))

        doy_table = np.array(doy_table, dtype=np.float32)
        logger.info(f"ابعاد doy_table ساخته شده: {doy_table.shape}")
        return doy_table
    except Exception as e:
        logger.error(f"خطا در خواندن calendar.txt: {e}")
        N_YEARS = len(year_list)
        return np.tile(np.arange(1, 367), (N_YEARS, 1)).astype(np.float32)


def create_memory_zarr_store_fast(n_stations, var_names):
    """
    ایجاد فروشگاه Zarr در حافظه با zarr_format=2
    """
    store = MemoryStore()
    root = zarr.open_group(store, mode='w', zarr_format=2)
    
    for name in var_names:
        arr = root.create(
            name,
            shape=(366, n_stations),
            dtype=np.float32,
            fill_value=np.nan,
            chunks=(366, min(500, n_stations)),
            overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['day', 'point']
    
    logger.info(f"فروشگاه Zarr در حافظه با {len(var_names)} متغیر ایجاد شد (zarr_format=2).")
    return root


def main():
    logger.info("=" * 60)
    logger.info("🧪 تست یک نقطه با فروشگاه سریع (zarr_format=2)")

    logger.info("مرحله 1: دریافت اطلاعات ایستگاه‌ها")
    n_stations, station_ids, lons, lats, elevs = get_station_info(ZARR_BASE)
    logger.info(f"تعداد کل ایستگاه‌ها: {n_stations}")

    logger.info("مرحله 2: ساخت جداول زمان اجرا")
    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    window_table = tables["window_table"]

    years_from_files = sorted({year for (year, month) in file_map.keys()})
    if not years_from_files:
        years_from_files = list(range(YEAR_START, YEAR_END + 1))
    year_list = years_from_files
    logger.info(f"سال‌های موجود: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    logger.info("مرحله 3: ساخت doy_table از calendar.txt")
    calendar_path = os.path.join(os.path.dirname(__file__), "..", "calendar.txt")
    if not os.path.exists(calendar_path):
        calendar_path = r"K:\Temp\needed\calendar.txt"
    logger.info(f"خواندن calendar.txt از: {calendar_path}")
    doy_table = build_doy_table_from_calendar_file(calendar_path, year_list)
    logger.info(f"ابعاد doy_table نهایی: {doy_table.shape}")

    logger.info("مرحله 4: ایجاد فروشگاه در حافظه (سریع با zarr_format=2)")
    root = create_memory_zarr_store_fast(n_stations, VAR_NAMES)

    logger.info("مرحله 5: شروع پردازش نقطه ۰ ...")
    block_start = 0
    block_end = 1
    block_idx = 0

    try:
        t0 = time.time()
        result = process_block(
            block_start=block_start,
            block_end=block_end,
            block_idx=block_idx,
            file_map=file_map,
            doy_table=doy_table,
            window_table=window_table,
            year_list=year_list,
            root=root,
            var_idx=VAR_INDEX_FOR_FIT,
            last_checkpoint_station=0
        )
        elapsed = time.time() - t0
        logger.info(f"✅ پردازش نقطه با موفقیت انجام شد. زمان: {elapsed:.2f} ثانیه")
    except Exception as e:
        logger.error(f"❌ خطا در پردازش: {e}", exc_info=True)
        raise

    logger.info("مرحله 6: بررسی داده‌های خروجی")
    ds = xr.open_zarr(root.store, consolidated=False, zarr_format=2)
    all_vars = list(ds.data_vars)

    for var_name in all_vars:
        data = ds[var_name].values
        if data.ndim == 2:
            point_data = data[:, 0]
        else:
            point_data = data
        valid = point_data[~np.isnan(point_data)]
        if len(valid) == 0:
            logger.info(f"   {var_name}: همه NaN")
        else:
            logger.info(f"   {var_name}:")
            logger.info(f"      تعداد معتبر: {len(valid)} از {len(point_data)}")
            logger.info(f"      میانگین: {np.mean(valid):.4f}")
            logger.info(f"      انحراف معیار: {np.std(valid):.4f}")
            logger.info(f"      حداقل: {np.min(valid):.4f}")
            logger.info(f"      حداکثر: {np.max(valid):.4f}")

    ds.close()

    try:
        clear_ds_cache()
    except:
        pass

    logger.info("\n✅ بررسی کامل شد. هیچ فایلی روی دیسک نوشته نشد.")


if __name__ == "__main__":
    main()