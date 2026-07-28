# -*- coding: utf-8 -*-
"""
read_month_files.py - بارگذاری فایل‌های ماهانه با کش دیسک و حافظه
(نسخه‌ی اصلاح‌شده با اسکیل صحیح برای همه متغیرهای دما)
"""

import os
import numpy as np
import xarray as xr
import shutil
from constants import CACHE_DIR, VARS, ZARR_BASE
from monitoring.logger import logger

_DS_CACHE = {}


def get_cached_or_load(year, month, var_idx, block_start, block_size, zarr_path):
    """بارگذاری داده با کش دیسک و اعمال اسکیل (تقسیم بر ۱۰) برای داده‌های int16"""
    cache_key = f"{year:04d}_{month:02d}_{var_idx}_{block_start}_{block_size}"

    if cache_key in _DS_CACHE:
        return _DS_CACHE[cache_key]

    cache_dir = os.path.join(CACHE_DIR, "zarr_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")

    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file, mmap_mode='r')
            # اگر داده int16 است، به float32 تبدیل و بر ۱۰ تقسیم کن
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 10.0
            _DS_CACHE[cache_key] = data
            return data
        except Exception:
            os.remove(cache_file)

    # بارگذاری از Zarr
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

        # ذخیره در کش به‌صورت int16 (برای صرفه‌جویی در فضا)
        if data.dtype != np.int16:
            data_for_cache = data.astype(np.int16)
        else:
            data_for_cache = data
        np.save(cache_file, data_for_cache)

        # اعمال اسکیل روی داده‌های برگشتی
        if data.dtype == np.int16:
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
    import os
    from monitoring.logger import logger

    n_vars = len(VARS)
    data_dict = {}

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
                        if False:
                            if key == (1397, 4):
                                print(f"\nRAW FROM ZARR {key} {var_name}")
                                print("dtype:", data.dtype)
                                print("shape:", data.shape)
                                print("min:", np.min(data))
                                print("max:", np.max(data))

                                bad = np.argwhere((data < -10000) | (data > 10000))
                                print("bad:", len(bad))
                                if len(bad):
                                    for d, s in bad[:10]:
                                        print(f"day={d}, station={s}, value={data[d, s]}")
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
                        var_name = VARS[v]
                        # ============================================================
                        # اصلاح اسکیل: برای متغیرهای دما، صرف‌نظر از dtype، بر ۱۰ تقسیم کن
                        # ============================================================
                        if var_name in ['tmin', 'tmean', 'tmax']:
                            data = data.astype(np.float32) / 10.0
                        else:
                            data = data.astype(np.float32)
                        combined[:, :, v] = data
                        if True:
                            if key == (1397, 4):
                                arr = combined[:, :, v]

                                print(f"\nAFTER ASSIGN {key} {var_name}")
                                print("min:", np.nanmin(arr))
                                print("max:", np.nanmax(arr))

                                bad = np.argwhere((arr < -1000) | (arr > 1000))
                                print("bad:", len(bad))

                                if len(bad):
                                    for d, s in bad[:10]:
                                        print(f"day={d}, station={s}, value={arr[d, s]}")
                if False:        
                 print("\n" + "="*80)
                 print(f"DEBUG MONTH: {key}")

                 for i, name in enumerate(["tmax", "tmean", "tmin"]):
                    arr = combined[:, :, i]

                    print(f"\n{name}")
                    print("shape :", arr.shape)
                    print("min   :", np.nanmin(arr))
                    print("max   :", np.nanmax(arr))

                    bad = np.argwhere((arr < -1000) | (arr > 1000))
                    print("bad values:", len(bad))

                    if len(bad):
                        print("First 10:")
                        for d, s in bad[:10]:
                            print(
                                f"day={d+1}, station={s}, value={arr[d,s]}"
                            )

                 print("="*80)        
                
                #data_dict[key] = combined
                data_dict[key] = combined.copy()

            except Exception as e:
                logger.warning(f"   ⚠️ Error reading {os.path.basename(file_path)}: {e}")
                continue

    logger.info(f"   ✅ Loaded {len(data_dict)} month-files")
    return data_dict


def clear_ds_cache():
    """پاک کردن کش حافظه‌ای و دیسک"""
    global _DS_CACHE
    _DS_CACHE.clear()
    cache_dir = os.path.join(CACHE_DIR, "zarr_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def clear_cache():
    clear_ds_cache()
