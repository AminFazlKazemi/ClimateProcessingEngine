#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_final_disk.py - تست نهایی با برازش همه متغیرها و ذخیره روی دیسک
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import zarr
import shutil

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
from zarr_schema import create_zarr_store, add_coords_and_metadata


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


def get_or_create_zarr_store(output_path, n_stations, station_ids, lons, lats, elevs):
    """ایجاد یا باز کردن فروشگاه Zarr روی دیسک"""
    if os.path.exists(output_path):
        logger.info(f"📂 Zarr موجود باز شد: {output_path}")
        # برای اطمینان، متادیتا را به‌روز می‌کنیم
        try:
            ds = xr.open_zarr(output_path)
            ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
            ds.to_zarr(output_path, mode='a', consolidated=True, zarr_format=2)
            ds.close()
        except Exception as e:
            logger.warning(f"به‌روزرسانی متادیتا ممکن نیست: {e}")
        return zarr.open_group(output_path, mode='a')
    else:
        logger.info(f"🆕 Zarr جدید ساخته شد: {output_path}")
        # ابتدا پوشه را خالی می‌کنیم (اگر وجود داشته باشد)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root = create_zarr_store(output_path, n_stations)
        # اضافه کردن متادیتا
        ds = xr.open_zarr(output_path)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.to_zarr(output_path, mode='w', consolidated=True, zarr_format=2)
        ds.close()
        return root


def main():
    logger.info("=" * 60)
    logger.info("🧪 تست نهایی با ذخیره روی دیسک (همه متغیرها)")

    # تعداد ایستگاه‌های تست (می‌توانید تغییر دهید)
    TEST_STATIONS = 10  # یا 100 برای تست بیشتر

    # دریافت اطلاعات ایستگاه‌ها
    n_stations, station_ids, lons, lats, elevs = get_station_info(ZARR_BASE)
    logger.info(f"تعداد کل ایستگاه‌ها: {n_stations}")
    actual_test = min(TEST_STATIONS, n_stations)
    logger.info(f"تعداد نقاط قابل پردازش: {actual_test}")

    # ساخت جداول زمان اجرا
    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    window_table = tables["window_table"]

    years_from_files = sorted({year for (year, month) in file_map.keys()})
    if not years_from_files:
        years_from_files = list(range(YEAR_START, YEAR_END + 1))
    year_list = years_from_files
    logger.info(f"سال‌های موجود: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    # ساخت doy_table از calendar.txt
    calendar_path = os.path.join(os.path.dirname(__file__), "..", "calendar.txt")
    if not os.path.exists(calendar_path):
        calendar_path = r"K:\Temp\needed\calendar.txt"
    logger.info(f"خواندن calendar.txt از: {calendar_path}")
    doy_table = build_doy_table_from_calendar_file(calendar_path, year_list)
    logger.info(f"ابعاد doy_table نهایی: {doy_table.shape}")

    # مسیر خروجی تست
    test_output = os.path.join(OUTPUT_DIR, "test_final_output.zarr")
    logger.info(f"خروجی در: {test_output}")

    # ایجاد فروشگاه روی دیسک
    root = get_or_create_zarr_store(test_output, n_stations, station_ids, lons, lats, elevs)

    # پردازش یک بلوک شامل ایستگاه‌های ۰ تا actual_test-1
    block_start = 0
    block_size = actual_test
    block_end = block_start + block_size
    block_idx = 0

    logger.info(f"شروع پردازش بلوک {block_idx}: ایستگاه‌های {block_start} تا {block_end-1}")

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
            var_idx=VAR_INDEX_FOR_FIT,  # این پارامتر در analyze_station بی‌تاثیر است
            last_checkpoint_station=0
        )
        elapsed = time.time() - t0
        logger.info(f"✅ پردازش بلوک با موفقیت انجام شد. زمان: {elapsed:.2f} ثانیه")
    except Exception as e:
        logger.error(f"❌ خطا در پردازش: {e}", exc_info=True)
        raise

    # بررسی خروجی برای هر سه متغیر
    logger.info("\n📊 بررسی داده‌های خروجی روی دیسک...")
    ds = xr.open_zarr(test_output, consolidated=False)

    # متغیرهای مورد انتظار برای هر سه متغیر
    var_names = ['tmin', 'tmean', 'tmax']
    dist_suffixes = ['best_dist', 'mean', 'std', 'skewness', 'median', 'count']
    for var in var_names:
        logger.info(f"\n🔍 بررسی متغیر {var}:")
        # آمار پایه
        for suffix in dist_suffixes:
            name = f"{var}_{suffix}"
            if name in ds:
                data = ds[name].values
                if data.ndim == 2:
                    point_data = data[:, 0]  # فقط نقطه اول
                else:
                    point_data = data
                valid = point_data[~np.isnan(point_data)]
                if len(valid) == 0:
                    logger.info(f"   {name}: همه NaN")
                else:
                    logger.info(f"   {name}: {len(valid)} معتبر از {len(point_data)}")
            else:
                logger.info(f"   {name}: وجود ندارد")

        # پارامترهای توزیع‌ها (نمونه)
        dist_names = ['normal', 'skewnormal', 'bimodal', 'pearson']
        for dist in dist_names:
            name = f"{var}_{dist}_p1"
            if name in ds:
                data = ds[name].values
                if data.ndim == 2:
                    point_data = data[:, 0]
                else:
                    point_data = data
                valid = point_data[~np.isnan(point_data)]
                if len(valid) == 0:
                    logger.info(f"   {name}: همه NaN")
                else:
                    logger.info(f"   {name}: {len(valid)} معتبر از {len(point_data)}")
            else:
                logger.info(f"   {name}: وجود ندارد")

    ds.close()

    try:
        clear_ds_cache()
    except:
        pass

    logger.info(f"\n✅ تست نهایی با موفقیت انجام شد. خروجی در: {test_output}")
    logger.info("📌 برای بررسی بیشتر از دستور زیر استفاده کنید:")
    logger.info(f"   ds = xr.open_zarr('{test_output}')")

if __name__ == "__main__":
    main()