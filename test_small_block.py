#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت تست برای پردازش فقط ۱۰ نقطه (ایستگاه) اول
با پشتیبانی از Zarr موجود و هماهنگی سال‌ها
"""

import os
import sys
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    YEAR_START, YEAR_END, VARS, VAR_INDEX_FOR_FIT,
    ZARR_BASE, OUTPUT_DIR, WINDOW_DAYS
)
from calendar_tables import build_doy_table_for_years   # ← تغییر
from runtime_tables import build_runtime_tables
from zarr_schema import create_zarr_store, add_coords_and_metadata
from orchestrator.process_block import process_block
from monitoring.logger import logger
from io_pipeline.read_month_files import clear_ds_cache


def get_station_info(zarr_base):
    import glob
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"هیچ فایل Zarr در {zarr_base} یافت نشد")
    ds = xr.open_zarr(zarr_files[0])
    n_stations = ds.sizes["point"]
    if "stationid" in ds:
        station_ids = ds["stationid"].values
    else:
        station_ids = np.arange(n_stations)
    lons = ds["lon"].values if "lon" in ds else np.full(n_stations, np.nan)
    lats = ds["lat"].values if "lat" in ds else np.full(n_stations, np.nan)
    elevs = ds["elev"].values if "elev" in ds else np.full(n_stations, np.nan)
    ds.close()
    return n_stations, station_ids, lons, lats, elevs


def get_or_create_zarr_store(output_path, n_stations, station_ids, lons, lats, elevs):
    if os.path.exists(output_path):
        logger.info(f"📂 Zarr موجود باز شد: {output_path}")
        ds = xr.open_zarr(output_path)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.to_zarr(output_path, mode="a", consolidated=True, zarr_format=2)
        ds.close()
        return xr.open_zarr(output_path)
    else:
        logger.info(f"🆕 Zarr جدید ساخته شد: {output_path}")
        root = create_zarr_store(output_path, n_stations)
        ds = xr.open_zarr(output_path)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.to_zarr(output_path, mode="w", consolidated=True, zarr_format=2)
        ds.close()
        return xr.open_zarr(output_path)


def main():
    TEST_POINTS = 10
    logger.info("=" * 60)
    logger.info(f"🚀 اجرای تست روی {TEST_POINTS} نقطه اول")

    n_stations, station_ids, lons, lats, elevs = get_station_info(ZARR_BASE)
    logger.info(f"تعداد کل ایستگاه‌ها: {n_stations}")

    actual_test = min(TEST_POINTS, n_stations)
    logger.info(f"تعداد نقاط قابل پردازش: {actual_test}")

    # ساخت جداول زمان اجرا (شامل file_map)
    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    window_table = tables["window_table"]

    # استخراج سال‌های واقعی از file_map
    years_from_files = sorted({year for (year, month) in file_map.keys()})
    if not years_from_files:
        years_from_files = list(range(YEAR_START, YEAR_END + 1))
    year_list = years_from_files
    logger.info(f"سال‌های موجود در فایل‌ها: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    # ساخت جدول تقویم بر اساس سال‌های واقعی (نه پیش‌فرض شمسی)
    doy_table, _ = build_doy_table_for_years(year_list)
    logger.info(f"ابعاد doy_table: {doy_table.shape}")

    test_output = os.path.join(OUTPUT_DIR, "test_output.zarr")
    root = get_or_create_zarr_store(test_output, n_stations, station_ids, lons, lats, elevs)
    logger.info(f"فروشگاه Zarr آماده است: {test_output}")

    # پردازش فقط ۱۰ نقطه اول (بلوک ۰)
    block_start = 0
    block_end = actual_test
    block_idx = 0

    logger.info(f"شروع پردازش بلوک {block_idx}: ایستگاه‌های {block_start} تا {block_end-1}")
    try:
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
        logger.info("✅ بلوک با موفقیت پردازش شد.")
    except Exception as e:
        logger.error(f"❌ خطا در پردازش بلوک: {e}")
        raise

    # به‌روزرسانی متادیتا (اختیاری)
    ds = xr.open_zarr(test_output)
    ds.attrs["last_updated"] = "تست با سال‌های هماهنگ انجام شد"
    ds.to_zarr(test_output, mode="a", consolidated=True, zarr_format=2)
    ds.close()

    try:
        clear_ds_cache()
    except:
        pass

    logger.info("=" * 60)
    logger.info(f"✅ تست با موفقیت انجام شد. خروجی در: {test_output}")
    logger.info("برای بررسی داده‌ها:")
    logger.info(f"    ds = xr.open_zarr('{test_output}')")


if __name__ == "__main__":
    main()