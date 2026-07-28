#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_logging.py - اصلاح خودکار فایل‌ها برای کاهش پیام‌های اضافی
"""

import os
import shutil
from pathlib import Path

# ============================================================
# مسیر ریشه (محل اسکریپت)
# ============================================================
BASE_DIR = Path(__file__).parent

# ============================================================
# محتوای جدید برای read_month_files.py
# ============================================================
READ_MONTH_CONTENT = '''# -*- coding: utf-8 -*-
"""
read_month_files.py - بارگذاری فایل‌های ماهانه با کش دیسک و حافظه
(نسخه‌ی بی‌صدا - فقط خطاها و خلاصه را چاپ می‌کند)
"""

import os
import numpy as np
import xarray as xr
import shutil
from constants import CACHE_DIR, VARS, ZARR_BASE
from monitoring.logger import logger

_DS_CACHE = {}

def get_cached_or_load(year, month, var_idx, block_start, block_size, zarr_path):
    cache_key = f"{year:04d}_{month:02d}_{var_idx}_{block_start}_{block_size}"

    if cache_key in _DS_CACHE:
        return _DS_CACHE[cache_key]

    cache_dir = os.path.join(CACHE_DIR, "zarr_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")

    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, mmap_mode='r')
            if var_idx == 1 and data.dtype == np.int16:
                data = data.astype(np.float32) / 10.0
            _DS_CACHE[cache_key] = data
            return data
        except Exception:
            os.remove(cache_file)

    var_name = VARS[var_idx]
    zarr_path = os.path.join(ZARR_BASE, f"{year:04d}_{month:02d}.zarr")
    if not os.path.exists(zarr_path):
        return None

    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        var_data = ds[var_name]
        dims = var_data.dims
        shape = var_data.shape

        if "point" in dims:
            point_axis = dims.index("point")
        else:
            point_axis = 0

        if point_axis == 0:
            start = min(block_start, shape[0])
            end = min(block_start + block_size, shape[0])
            if start >= end:
                ds.close()
                return None
            data = var_data.values[start:end, ...]
        else:
            indices = list(range(block_start, min(block_start + block_size, shape[point_axis])))
            data = var_data.isel({point_axis: indices}).values
            if data.ndim >= 2:
                data = data.T

        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)

        ds.close()

        if data.dtype != np.int16:
            data_for_cache = data.astype(np.int16)
        else:
            data_for_cache = data

        np.save(cache_file, data_for_cache)

        if var_idx == 1:
            data = data.astype(np.float32) / 10.0

        _DS_CACHE[cache_key] = data
        return data
    except Exception as e:
        if block_start == 0 and year == 1370 and month == 1:
            logger.error(f"Exception loading {year}-{month}: {e}")
        return None

def read_month_files(block_start, block_size, file_map, year_list, var_idx=None):
    """
    خواندن داده‌های ماهانه برای یک بازه از نقاط
    حالا همه متغیرها را بارگذاری می‌کند (var_idx نادیده گرفته می‌شود)
    """
    from constants import VARS
    import numpy as np
    import xarray as xr

    n_vars = len(VARS)
    data_dict = {}

    # فقط یک پیام خلاصه در ابتدا
    logger.info(f"   📂 Loading {len(year_list)} years, block_size={block_size}")

    for year in year_list:
        for month in range(1, 13):
            key = (year, month)
            if key not in file_map:
                continue

            file_path = file_map[key]

            try:
                ds = xr.open_zarr(file_path)
                dims = list(ds.dims.keys())
                if len(dims) < 2:
                    ds.close()
                    continue

                point_dim = dims[1]

                # خواندن همه متغیرها
                var_data = {}
                for v, var_name in enumerate(VARS):
                    if var_name in ds:
                        data = ds[var_name].isel({point_dim: slice(block_start, block_start + block_size)}).values
                        var_data[v] = data
                    else:
                        var_data[v] = None

                ds.close()

                if not any(v is not None for v in var_data.values()):
                    continue

                # ترکیب متغیرها در یک آرایه ۳بعدی
                first_var = next(v for v in var_data.values() if v is not None)
                days, points = first_var.shape
                combined = np.full((days, points, n_vars), np.nan, dtype=np.float32)
                for v, data in var_data.items():
                    if data is not None:
                        if data.dtype == np.int16:
                            data = data.astype(np.float32)
                            if v == 1:
                                data = data / 10.0
                        else:
                            data = data.astype(np.float32)
                        combined[:, :, v] = data

                data_dict[key] = combined

            except Exception as e:
                logger.warning(f"   ⚠️ Error reading {os.path.basename(file_path)}: {e}")
                continue

    logger.info(f"   ✅ Loaded {len(data_dict)} month-files")
    return data_dict

def clear_ds_cache():
    global _DS_CACHE
    _DS_CACHE.clear()
    cache_dir = os.path.join(CACHE_DIR, "zarr_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)

def clear_cache():
    clear_ds_cache()
'''

# ============================================================
# محتوای جدید برای assemble_block.py
# ============================================================
ASSEMBLE_BLOCK_CONTENT = '''# -*- coding: utf-8 -*-
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
    logger.info(f"   ✅ Assembled block: {total_valid:,} valid values")
    return block_data
'''

# ============================================================
# توابع کمکی
# ============================================================
def find_file(filename, search_dirs=None):
    """پیدا کردن فایل در دایرکتوری‌های مشخص"""
    if search_dirs is None:
        search_dirs = [BASE_DIR]
    
    for base in search_dirs:
        for root, dirs, files in os.walk(base):
            if filename in files:
                return Path(root) / filename
    return None

def replace_file(filepath, new_content, backup_suffix='.bak'):
    """جایگزینی فایل با محتوای جدید و گرفتن پشتیبان"""
    if not filepath or not filepath.exists():
        return False
    
    # پشتیبان‌گیری
    backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)
    shutil.copy2(filepath, backup_path)
    print(f"📁 Backup saved: {backup_path}")
    
    # نوشتن محتوای جدید
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Updated: {filepath}")
    return True

# ============================================================
# تابع اصلی
# ============================================================
def main():
    print("=" * 70)
    print("🔧 Fix logging - Reducing verbose output")
    print("=" * 70)
    
    # پیدا کردن فایل‌ها
    read_month_path = find_file("read_month_files.py", [
        BASE_DIR / "io_pipeline",
        BASE_DIR
    ])
    
    assemble_path = find_file("assemble_block.py", [
        BASE_DIR / "io_pipeline",
        BASE_DIR
    ])
    
    # جایگزینی فایل‌ها
    success = True
    
    if read_month_path:
        print(f"\n📂 Found: {read_month_path}")
        if replace_file(read_month_path, READ_MONTH_CONTENT):
            print("   ✅ read_month_files.py updated successfully")
        else:
            print("   ❌ Failed to update read_month_files.py")
            success = False
    else:
        print("\n⚠️ read_month_files.py not found!")
        success = False
    
    if assemble_path:
        print(f"\n📂 Found: {assemble_path}")
        if replace_file(assemble_path, ASSEMBLE_BLOCK_CONTENT):
            print("   ✅ assemble_block.py updated successfully")
        else:
            print("   ❌ Failed to update assemble_block.py")
            success = False
    else:
        print("\n⚠️ assemble_block.py not found!")
        success = False
    
    # نتیجه‌ی نهایی
    print("\n" + "=" * 70)
    if success:
        print("✅ All files updated successfully!")
        print("📌 You can now run main.py with reduced logging.")
    else:
        print("⚠️ Some files could not be updated. Please check manually.")
    print("=" * 70)

if __name__ == "__main__":
    main()