#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست یک نقطه با استفاده از Zarr موجود (اگر وجود داشته باشد)
"""

import os
import sys
import numpy as np
import xarray as xr
import zarr

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    YEAR_START, YEAR_END, VARS, VAR_INDEX_FOR_FIT,
    ZARR_BASE, OUTPUT_DIR, WINDOW_DAYS
)
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
    """
    اگر Zarr وجود داشته باشد، آن را باز می‌کند.
    در غیر این صورت، یک Zarr جدید ایجاد می‌کند.
    برمی‌گرداند: zarr.group (برای استفاده در process_block)
    """
    if os.path.exists(output_path):
        logger.info(f"📂 Zarr موجود باز شد: {output_path}")
        # باز کردن گروه Zarr به صورت مستقیم (برای process_block)
        root = zarr.open_group(output_path, mode='a')
        # متادیتا را نیز به‌روز می‌کنیم (اختیاری)
        try:
            ds = xr.open_zarr(output_path)
            ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
            ds.to_zarr(output_path, mode='a', consolidated=True, zarr_format=2)
            ds.close()
        except Exception as e:
            logger.warning(f"به‌روزرسانی متادیتا ممکن نیست: {e}")
        return root
    else:
        logger.info(f"🆕 Zarr جدید ساخته شد: {output_path}")
        root = create_zarr_store(output_path, n_stations)
        # اضافه کردن متادیتا
        ds = xr.open_zarr(output_path)
        ds = add_coords_and_metadata(ds, station_ids, lons, lats, elevs)
        ds.to_zarr(output_path, mode='w', consolidated=True, zarr_format=2)
        ds.close()
        return root


def main():
    logger.info("=" * 60)
    logger.info("🧪 تست یک نقطه (ایستگاه اول) - با استفاده از Zarr موجود")

    n_stations, station_ids, lons, lats, elevs = get_station_info(ZARR_BASE)
    logger.info(f"تعداد کل ایستگاه‌ها: {n_stations}")

    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    window_table = tables["window_table"]

    years_from_files = sorted({year for (year, month) in file_map.keys()})
    if not years_from_files:
        years_from_files = list(range(YEAR_START, YEAR_END + 1))
    year_list = years_from_files
    logger.info(f"سال‌های موجود: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    # ========== doy_table ساختگی (۳۶۶ روز) ==========
    N_YEARS = len(year_list)
    N_DAYS = 366
    doy_table = np.tile(np.arange(1, N_DAYS + 1), (N_YEARS, 1))
    logger.info(f"ابعاد doy_table (ساختگی): {doy_table.shape}")

    # مسیر خروجی
    test_output = os.path.join(OUTPUT_DIR, "test_one_point.zarr")

    # دریافت یا ایجاد فروشگاه
    root = get_or_create_zarr_store(test_output, n_stations, station_ids, lons, lats, elevs)
    logger.info(f"فروشگاه Zarr آماده است: {test_output}")

    # پردازش فقط نقطه ۰
    block_start = 0
    block_end = 1
    block_idx = 0

    logger.info("شروع پردازش نقطه ۰ ...")
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
        logger.info("✅ پردازش نقطه با موفقیت انجام شد.")
    except Exception as e:
        logger.error(f"❌ خطا: {e}")
        raise

    # ============ بررسی داده‌ها (همانند قبل) ============
    logger.info("\n" + "=" * 60)
    logger.info("📊 بررسی داده‌های خروجی برای نقطه ۰")

    ds = xr.open_zarr(test_output)
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
            if len(point_data) <= 366:
                logger.info(f"      ۱۰ روز اول: {point_data[:10].tolist()}")

    ds.close()
    logger.info(f"\n✅ بررسی کامل شد. فایل در {test_output} باقی می‌ماند.")

    try:
        clear_ds_cache()
    except:
        pass


if __name__ == "__main__":
    main()