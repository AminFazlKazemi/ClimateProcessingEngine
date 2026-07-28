# -*- coding: utf-8 -*-
"""
assemble_block.py - مونتاژ داده‌های ماهانه در یک بلوک
(نسخه‌ی بی‌صدا - فقط خلاصه را چاپ می‌کند)
"""

import numpy as np
from monitoring.logger import logger

def assemble_block(data_dict, doy_table, block_size, year_list, var_idx=None):
    """
    مونتاژ داده‌های ماهانه در یک بلوک (block)
    داده‌های ورودی از فایل‌های Zarr شامل همه متغیرها هستند.
    خروجی: float32 با ابعاد (block_size, N_YEARS, N_DAYS, n_vars)
    """
    N_YEARS = len(year_list)
    N_DAYS = 366
    n_vars = 3  # tmax, tmean, tmin

    # ایجاد آرایه خالی با NaN از نوع float32
    block_data = np.full((block_size, N_YEARS, N_DAYS, n_vars), np.nan, dtype=np.float32)

    if not data_dict:
        logger.warning("   ⚠️ data_dict is empty. Output will be all NaN.")
        return block_data

    # جدول تقریبی offset روزهای شروع هر ماه (برای سال عادی)
    month_offsets = [0, 31, 62, 93, 124, 155, 186, 216, 246, 276, 306, 336]

    total_valid = 0
    for year_idx, year in enumerate(year_list):
        for month in range(1, 13):
            key = (year, month)
            if key not in data_dict:
                continue

            month_data = data_dict[key]  # shape: (days_in_month, block_size, n_vars)

            # ============================================================
            # بلوک دیباگ - غیرفعال (if False)
            # ============================================================
            if False:  # غیرفعال شده
                if year == 1397 and month == 4:
                    print("\n" + "="*80)
                    print("MONTH_DATA BEFORE COPY")
                    print("tmax :", np.nanmin(month_data[:,:,0]), np.nanmax(month_data[:,:,0]))
                    print("tmean:", np.nanmin(month_data[:,:,1]), np.nanmax(month_data[:,:,1]))
                    print("tmin :", np.nanmin(month_data[:,:,2]), np.nanmax(month_data[:,:,2]))
                    bad = np.argwhere((month_data < -1000) | (month_data > 1000))
                    print("bad:", len(bad))
                    if len(bad):
                        print(bad[:20])

            if month_data is None or month_data.size == 0:
                continue

            if month_data.ndim != 3:
                continue

            n_days = month_data.shape[0]
            start_idx = month_offsets[month-1]

            # قرار دادن داده‌ها در آرایه اصلی
            for d in range(min(n_days, N_DAYS - start_idx)):
                block_data[:, year_idx, start_idx + d, :] = month_data[d, :, :]
                total_valid += block_size * n_vars

    total_valid = np.count_nonzero(~np.isnan(block_data))

    # ============================================================
    # بلوک دیباگ - غیرفعال (if False)
    # ============================================================
    if False:  # غیرفعال شده
        print("\n" + "=" * 80)
        print("ASSEMBLE DEBUG")
        print("=" * 80)

        for i, name in enumerate(["tmax", "tmean", "tmin"]):
            arr = block_data[:, :, :, i]
            print(f"\n{name}")
            print("dtype :", arr.dtype)
            print("shape :", arr.shape)
            print("min   :", np.nanmin(arr))
            print("max   :", np.nanmax(arr))
            bad = np.argwhere((arr < -1000) | (arr > 1000))
            print("bad values:", len(bad))
            if len(bad):
                print("First 10 bad values:")
                for s, y, d in bad[:10]:
                    print(f"station={s}, year={year_list[y]}, day={d+1}, value={arr[s,y,d]}")
        print("=" * 80)

    logger.info(f"   ✅ Assembled block: {total_valid:,} valid values")
    return block_data