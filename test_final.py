#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_final_full_vars.py
تست کامل با ۵ سال و ۱ ایستگاه و همه ۹۳ متغیر خروجی
"""

import os
import sys
import time
import numpy as np
import xarray as xr
import zarr
from zarr.storage import MemoryStore
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from constants import (
    YEAR_START, YEAR_END, VARS, VAR_INDEX_FOR_FIT,
    ZARR_BASE, WINDOW_DAYS
)
from runtime_tables import build_runtime_tables
from orchestrator.process_block import process_block
from io_pipeline.read_month_files import clear_ds_cache
from result_pipeline.write_block import VAR_NAMES


def get_station_info(zarr_base):
    import glob
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    if not zarr_files:
        raise FileNotFoundError(f"هیچ فایل Zarr در {zarr_base} یافت نشد")
    ds = xr.open_zarr(zarr_files[0])
    n_stations = ds.sizes["point"]
    ds.close()
    return n_stations


def build_doy_table_simple(year_list):
    """ساخت doy_table ساده برای سال‌های مشخص"""
    N_YEARS = len(year_list)
    N_DAYS = 366
    doy_table = np.tile(np.arange(1, N_DAYS + 1), (N_YEARS, 1)).astype(np.float32)
    return doy_table


def create_memory_zarr_store_full(n_stations):
    """
    ایجاد فروشگاه Zarr در حافظه با همه ۹۳ متغیر خروجی
    (مشابه zarr_schema.create_zarr_store)
    """
    store = MemoryStore()
    root = zarr.open_group(store, mode='w', zarr_format=2)
    
    print(f"📦 ایجاد {len(VAR_NAMES)} متغیر در حافظه...")
    for name in VAR_NAMES:
        root.create(
            name,
            shape=(366, n_stations),
            dtype=np.float32,
            fill_value=np.nan,
            chunks=(366, min(500, n_stations)),
            overwrite=True
        )
    
    print(f"✅ فروشگاه Zarr با {len(VAR_NAMES)} متغیر ایجاد شد.")
    return root


def main():
    print("=" * 70)
    print("⚡ تست با ۵ سال داده و همه ۹۳ متغیر خروجی")
    print("=" * 70)

    start_time = time.time()

    # ۱. دریافت تعداد ایستگاه‌ها
    n_stations = get_station_info(ZARR_BASE)
    print(f"📊 تعداد کل ایستگاه‌ها: {n_stations:,}")

    # ۲. ساخت جداول زمان اجرا
    tables = build_runtime_tables(ZARR_BASE)
    file_map = tables["file_map"]
    window_table = tables["window_table"]

    # ۳. محدود کردن به ۵ سال اول
    all_years = sorted({year for (year, month) in file_map.keys()})
    if not all_years:
        all_years = list(range(YEAR_START, YEAR_END + 1))
    year_list = all_years[:5]
    print(f"📅 سال‌های انتخابی: {year_list[0]}–{year_list[-1]} ({len(year_list)} سال)")

    # ۴. ساخت doy_table ساده
    doy_table = build_doy_table_simple(year_list)
    print(f"📋 doy_table shape: {doy_table.shape}")

    # ۵. ایجاد فروشگاه با همه متغیرها
    root = create_memory_zarr_store_full(n_stations)

    # ۶. پردازش فقط ۱ ایستگاه
    block_start = 0
    block_size = 1
    block_end = block_start + block_size
    block_idx = 0

    print(f"\n🚀 شروع پردازش ایستگاه ۰ با {len(year_list)} سال...")

    t0 = time.time()
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
        elapsed = time.time() - t0
        print(f"✅ پردازش کامل شد! زمان: {elapsed:.2f} ثانیه")
    except Exception as e:
        print(f"❌ خطا: {e}")
        raise

    # ۷. بررسی سریع خروجی
    print("\n📊 بررسی داده‌های خروجی برای ایستگاه ۰:")
    ds = xr.open_zarr(root.store, consolidated=False, zarr_format=2)

    # بررسی tmean_mean
    mean_data = ds["tmean_mean"].values[:, 0]
    valid_mean = mean_data[~np.isnan(mean_data)]
    print(f"   tmean_mean: {len(valid_mean)} روز معتبر از 366")
    if len(valid_mean) > 0:
        print(f"      میانگین: {np.mean(valid_mean):.2f} °C")
        print(f"      محدوده: {np.min(valid_mean):.2f} تا {np.max(valid_mean):.2f} °C")

    # بررسی tmin_mean (برای اطمینان از اینکه tmin هم پردازش شده)
    mean_data_tmin = ds["tmin_mean"].values[:, 0]
    valid_mean_tmin = mean_data_tmin[~np.isnan(mean_data_tmin)]
    print(f"   tmin_mean: {len(valid_mean_tmin)} روز معتبر از 366")
    if len(valid_mean_tmin) > 0:
        print(f"      میانگین: {np.mean(valid_mean_tmin):.2f} °C")

    # بررسی best_dist برای tmean
    best_data = ds["tmean_best_dist"].values[:, 0]
    valid_best = best_data[~np.isnan(best_data)]
    print(f"   tmean_best_dist: {len(valid_best)} روز معتبر از 366")
    if len(valid_best) > 0:
        unique, counts = np.unique(valid_best, return_counts=True)
        dist_names = {0: "Normal", 1: "SkewNormal", 2: "Bimodal", 3: "Pearson"}
        for code, count in zip(unique, counts):
            name = dist_names.get(int(code), f"Unknown({int(code)})")
            print(f"      {name}: {count} روز ({count/len(valid_best)*100:.1f}%)")

    ds.close()

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ تست با ۵ سال و همه متغیرها با موفقیت کامل شد! (کل زمان: {total_time:.2f} ثانیه)")
    print("📌 هیچ فایلی روی دیسک نوشته نشد.")
    print("=" * 70)

    clear_ds_cache()


if __name__ == "__main__":
    main()