#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_adapter.py
================================================================================
لایه انتزاع داده برای پشتیبانی از فرمت‌های ایستگاهی و شبکه‌ای (Gridded)
نسخه ۴.۷ – کش فقط برای بلوک‌های بدون کش ساخته می‌شود (با بررسی وجود فایل)
================================================================================
"""

import os
import glob
import numpy as np
import xarray as xr
import pickle
import hashlib
import gzip
from typing import Optional, List, Dict, Tuple, Any, Union
from constants import VARS


# ============================================================================
# ۱. مدیریت کش دیسک (با ذخیره‌ی شرطی)
# ============================================================================

class DiskCache:
    """
    کش دیسک با قابلیت جستجوی چندین کلید و ذخیره‌ی جدید فقط در صورت نیاز.
    
    ویژگی‌ها:
        - جستجوی کش با اولویت: block_size=1000, 2000, 5000 با hash="all" و سپس hash فعلی
        - ذخیره‌ی کش جدید فقط در صورتی که فایل کش وجود نداشته باشد
        - مدیریت حجم کش (حداکثر ۲۰ گیگابایت)
    """

    def __init__(self, cache_dir: str = "./cache", max_size_gb: int = 20):
        """
        پارامترها:
            cache_dir: مسیر پوشه‌ی کش
            max_size_gb: حداکثر حجم کش به گیگابایت
        """
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_gb * 1024**3
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self,
        block_start: int,
        block_size: int,
        year: int,
        month: int,
        var: Any,
        sample_hash: str = "all"
    ) -> str:
        """
        ساخت کلید منحصربه‌فرد برای هر ترکیب پارامتر.
        
        پارامترها:
            block_start: اندیس شروع بلوک
            block_size: اندازه‌ی بلوک
            year: سال
            month: ماه
            var: شناسه‌ی متغیر (عدد یا رشته)
            sample_hash: هش نمونه‌برداری (پیش‌فرض "all")
        
        بازگشت:
            رشته‌ی هش MD5
        """
        key_str = f"{block_start}_{block_size}_{year}_{month}_{var}_{sample_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> str:
        """
        مسیر کامل فایل کش.
        
        پارامترها:
            key: کلید کش
        
        بازگشت:
            مسیر فایل با پسوند .pkl.gz
        """
        return os.path.join(self.cache_dir, f"{key}.pkl.gz")

    def _load_cached_data(self, path: str) -> Optional[np.ndarray]:
        """
        بارگذاری داده از فایل کش (با پشتیبانی از gzip و غیر gzip).
        
        پارامترها:
            path: مسیر فایل کش
        
        بازگشت:
            آرایه‌ی numpy در صورت موفقیت، در غیر این صورت None
        """
        if not os.path.exists(path):
            return None
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None

    def get(
        self,
        block_start: int,
        block_size: int,
        year: int,
        month: int,
        var: Any,
        sample_hash: str = "all"
    ) -> Optional[np.ndarray]:
        """
        جستجوی کش با اولویت:
        ۱. block_size=1000, 2000, 5000 با sample_hash="all"
        ۲. block_size=1000, 2000, 5000 با sample_hash فعلی
        ۳. block_size و block_start فعلی با sample_hash فعلی
        
        اگر کشی پیدا شد، آن را برمی‌گرداند (و هرگز کپی نمی‌کند).
        
        پارامترها:
            block_start: اندیس شروع بلوک
            block_size: اندازه‌ی بلوک
            year: سال
            month: ماه
            var: شناسه‌ی متغیر
            sample_hash: هش نمونه‌برداری
        
        بازگشت:
            آرایه‌ی numpy در صورت پیدا شدن کش، در غیر این صورت None
        """
        # لیست تمام ترکیب‌های (block_start, block_size, sample_hash) به ترتیب اولویت
        search_order = []
        
        # تطبیق block_start برای کش‌های قدیمی
        adjusted_start_1000 = (block_start // 1000) * 1000
        adjusted_start_2000 = (block_start // 2000) * 2000
        adjusted_start_5000 = (block_start // 5000) * 5000
        
        # اولویت ۱ تا ۳: کش‌های قدیمی با sample_hash="all"
        search_order.append((adjusted_start_1000, 1000, "all"))
        search_order.append((adjusted_start_2000, 2000, "all"))
        search_order.append((adjusted_start_5000, 5000, "all"))
        
        # اولویت ۴ تا ۶: کش‌های جدید با sample_hash فعلی (اگر با "all" فرق داره)
        if sample_hash != "all":
            search_order.append((adjusted_start_1000, 1000, sample_hash))
            search_order.append((adjusted_start_2000, 2000, sample_hash))
            search_order.append((adjusted_start_5000, 5000, sample_hash))
        
        # اولویت ۷: block_start و block_size فعلی با sample_hash جدید
        search_order.append((block_start, block_size, sample_hash))
        
        # جستجو
        for bs_start, bs_size, bs_hash in search_order:
            key = self._get_cache_key(bs_start, bs_size, year, month, var, bs_hash)
            path = self._get_cache_path(key)
            data = self._load_cached_data(path)
            if data is not None:
                print(f"   ✅ کش استفاده شد: {path}")
                print(f"      block_start={bs_start}, block_size={bs_size}, hash={bs_hash}")
                return data
        
        return None

    def set(
        self,
        block_start: int,
        block_size: int,
        year: int,
        month: int,
        var: Any,
        data: np.ndarray,
        sample_hash: str = "all"
    ) -> None:
        """
        ذخیره‌ی داده در کش با کلید اصلی (block_start, block_size, sample_hash فعلی).
        
        ⚠️ این متد فقط زمانی ذخیره می‌کند که فایل کش از قبل وجود نداشته باشد.
        اگر فایل کش از قبل وجود داشته باشد، هیچ عملی انجام نمی‌دهد (از duplicate جلوگیری می‌کند).
        
        پارامترها:
            block_start: اندیس شروع بلوک
            block_size: اندازه‌ی بلوک
            year: سال
            month: ماه
            var: شناسه‌ی متغیر
            data: آرایه‌ی numpy برای ذخیره
            sample_hash: هش نمونه‌برداری
        """
        # کلید و مسیر فایل کش را محاسبه کن
        key = self._get_cache_key(block_start, block_size, year, month, var, sample_hash)
        path = self._get_cache_path(key)
        
        # ============================================================
        # ✅ اگر فایل کش از قبل وجود دارد، هیچ کاری نکن (از duplicate جلوگیری کن)
        # ============================================================
        if os.path.exists(path):
            print(f"   ⏭️ کش از قبل وجود دارد، ذخیره نشد: {path}")
            return
        
        # مدیریت حجم کش
        self._enforce_cache_limit()
        
        # ذخیره با فشرده‌سازی gzip
        with gzip.open(path, "wb", compresslevel=6) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"   💾 کش جدید ذخیره شد: {path}")

    def _enforce_cache_limit(self) -> None:
        """
        مدیریت حجم کش: اگر از حد مجاز بیشتر شد، قدیمی‌ترین فایل‌ها حذف می‌شوند.
        """
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in os.listdir(self.cache_dir)
            if f.endswith((".pkl", ".pkl.gz"))
        )
        if total_size > self.max_size_bytes:
            files = sorted(
                [
                    os.path.join(self.cache_dir, f)
                    for f in os.listdir(self.cache_dir)
                    if f.endswith((".pkl", ".pkl.gz"))
                ],
                key=os.path.getmtime
            )
            while files and total_size > self.max_size_bytes:
                removed = files.pop(0)
                os.remove(removed)
                total_size = sum(
                    os.path.getsize(os.path.join(self.cache_dir, f))
                    for f in os.listdir(self.cache_dir)
                    if f.endswith((".pkl", ".pkl.gz"))
                )


# ============================================================================
# ۲. توابع شناسایی خودکار فرمت داده و استخراج اطلاعات
# ============================================================================

def detect_data_format(zarr_path: str) -> Optional[str]:
    """
    تشخیص فرمت داده‌های Zarr: 'station' یا 'gridded' یا None
    
    پارامترها:
        zarr_path: مسیر فایل Zarr نمونه
    
    بازگشت:
        'station' برای داده‌های ایستگاهی
        'gridded' برای داده‌های شبکه‌ای
        None در صورت عدم تشخیص
    """
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        dims = list(ds.dims)
        has_point = "point" in dims
        has_stationid = "stationid" in ds.data_vars or "stationid" in ds.coords
        has_grid = all(d in dims for d in ["latitude", "longitude", "time"])
        ds.close()
        if has_point and has_stationid:
            return "station"
        elif has_grid:
            return "gridded"
        else:
            return None
    except Exception:
        return None


def get_station_info(zarr_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    استخراج شناسه‌ها، طول، عرض و ارتفاع ایستگاه‌ها از یک فایل Zarr نمونه.
    
    پارامترها:
        zarr_path: مسیر فایل Zarr نمونه
    
    بازگشت:
        (station_ids, lats, lons, elevs)
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    
    # استخراج شناسه‌های ایستگاه
    if "stationid" in ds.coords:
        station_ids = ds["stationid"].values
    else:
        station_ids = ds["stationid"].values if "stationid" in ds.data_vars else None
    
    # استخراج عرض جغرافیایی
    if "lat" in ds.coords:
        lats = ds["lat"].values
    else:
        lats = ds["lat"].values if "lat" in ds.data_vars else None
    
    # استخراج طول جغرافیایی
    if "lon" in ds.coords:
        lons = ds["lon"].values
    else:
        lons = ds["lon"].values if "lon" in ds.data_vars else None
    
    # استخراج ارتفاع
    if "elev" in ds.coords:
        elevs = ds["elev"].values
    elif "elev" in ds.data_vars:
        elevs = ds["elev"].values
    else:
        elevs = np.zeros_like(station_ids)
    
    ds.close()
    return station_ids, lats, lons, elevs


def get_grid_info(zarr_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    استخراج طول و عرض شبکه از یک فایل Zarr نمونه.
    
    پارامترها:
        zarr_path: مسیر فایل Zarr نمونه
    
    بازگشت:
        (latitudes, longitudes)
    """
    ds = xr.open_zarr(zarr_path, consolidated=False)
    
    if "latitude" in ds.coords:
        lat = ds["latitude"].values
    else:
        lat = ds["latitude"].values if "latitude" in ds.data_vars else None
    
    if "longitude" in ds.coords:
        lon = ds["longitude"].values
    else:
        lon = ds["longitude"].values if "longitude" in ds.data_vars else None
    
    ds.close()
    return lat, lon


# ============================================================================
# ۳. کلاس‌های پایه و پیاده‌سازی‌های Adapter
# ============================================================================

class BaseDataAdapter:
    """کلاس پایه برای تمام Adapterها"""

    def __init__(
        self,
        zarr_base: str,
        year_list: List[int],
        cache_enabled: bool = True
    ):
        """
        پارامترها:
            zarr_base: مسیر پایه‌ی فایل‌های Zarr
            year_list: لیست سال‌ها
            cache_enabled: فعال/غیرفعال کردن کش
        """
        self.zarr_base = zarr_base
        self.year_list = year_list
        self._file_map: Optional[Dict[Tuple[int, int], str]] = None
        self._n_points: Optional[int] = None
        self._selected_indices: Optional[np.ndarray] = None
        self.max_points: Optional[int] = None
        self.cache = DiskCache() if cache_enabled else None

    @property
    def n_points(self) -> int:
        """تعداد نقاط (ایستگاه‌ها یا پیکسل‌های شبکه)"""
        if self._n_points is None:
            raise ValueError("n_points not initialized")
        return self._n_points

    @property
    def file_map(self) -> Dict[Tuple[int, int], str]:
        """نگاشت (سال, ماه) → مسیر فایل Zarr"""
        if self._file_map is None:
            self._build_file_map()
        return self._file_map

    def _build_file_map(self) -> None:
        """ساخت نگاشت (سال, ماه) → مسیر فایل Zarr از فایل‌های موجود"""
        self._file_map = {}
        zarr_files = glob.glob(os.path.join(self.zarr_base, "*.zarr"))
        for f in zarr_files:
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    self._file_map[(year, month)] = f
                except ValueError:
                    continue

    def _get_sample_hash(self) -> str:
        """
        محاسبه هش از اندیس‌های انتخاب‌شده (برای تشخیص تغییر sample).
        اگر نمونه‌برداری انجام نشده باشد، "all" برمی‌گرداند.
        """
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            return hashlib.md5(self._selected_indices.tobytes()).hexdigest()[:8]
        return "all"

    def load_block(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int,
        var_idx: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از داده‌های یک متغیر خاص.
        
        پارامترها:
            block_start: اندیس شروع بلوک
            block_size: اندازه‌ی بلوک
            year_idx: اندیس سال در year_list
            month: ماه (۱-۱۲)
            var_idx: اندیس متغیر در VARS
        
        بازگشت:
            آرایه‌ی numpy با ابعاد (days, block_size) یا None در صورت خطا
        """
        raise NotImplementedError

    def load_block_all_vars(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از همه متغیرها به‌صورت همزمان.
        
        پارامترها:
            block_start: اندیس شروع بلوک
            block_size: اندازه‌ی بلوک
            year_idx: اندیس سال در year_list
            month: ماه (۱-۱۲)
        
        بازگشت:
            آرایه‌ی numpy با ابعاد (days, block_size, n_vars) یا None در صورت خطا
        """
        raise NotImplementedError

    def get_coords(self) -> Dict[str, Any]:
        """
        برگرداندن مختصات نقاط (ایستگاه‌ها یا پیکسل‌های شبکه).
        
        بازگشت:
            دیکشنری شامل 'stationid', 'lat', 'lon', 'elev' و غیره
        """
        raise NotImplementedError


class StationDataAdapter(BaseDataAdapter):
    """
    Adapter برای داده‌های ایستگاهی (Station-based)
    """

    def __init__(
        self,
        zarr_base: str,
        year_list: List[int],
        cache_enabled: bool = True,
        max_points: int = 40000
    ):
        """
        پارامترها:
            zarr_base: مسیر پایه‌ی فایل‌های Zarr
            year_list: لیست سال‌ها
            cache_enabled: فعال/غیرفعال کردن کش
            max_points: حداکثر تعداد نقاط (برای نمونه‌برداری)
        """
        super().__init__(zarr_base, year_list, cache_enabled)
        self.max_points = max_points
        self._station_ids: Optional[np.ndarray] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        self._elevs: Optional[np.ndarray] = None
        self._initialize_from_sample()

    def _initialize_from_sample(self) -> None:
        """
        بارگذاری اطلاعات ایستگاه‌ها از اولین فایل Zarr موجود و اعمال نمونه‌برداری.
        """
        zarr_files = glob.glob(os.path.join(self.zarr_base, "*.zarr"))
        if not zarr_files:
            raise FileNotFoundError(f"No Zarr files found in {self.zarr_base}")

        ds = xr.open_zarr(zarr_files[0], consolidated=False)
        total_points = ds.sizes["point"]

        # استخراج اطلاعات
        if "stationid" in ds.coords:
            all_station_ids = ds["stationid"].values
        else:
            all_station_ids = ds["stationid"].values if "stationid" in ds.data_vars else np.arange(total_points)

        if "lat" in ds.coords:
            all_lats = ds["lat"].values
        else:
            all_lats = ds["lat"].values if "lat" in ds.data_vars else np.zeros(total_points)

        if "lon" in ds.coords:
            all_lons = ds["lon"].values
        else:
            all_lons = ds["lon"].values if "lon" in ds.data_vars else np.zeros(total_points)

        if "elev" in ds.coords:
            all_elevs = ds["elev"].values
        elif "elev" in ds.data_vars:
            all_elevs = ds["elev"].values
        else:
            all_elevs = np.zeros(total_points)

        ds.close()

        # نمونه‌برداری تصادفی (با seed ثابت برای تکرارپذیری)
        if total_points > self.max_points:
            np.random.seed(42)
            indices = np.random.choice(total_points, size=self.max_points, replace=False)
            indices = np.sort(indices)
            self._selected_indices = indices
            self._n_points = self.max_points
            self._station_ids = all_station_ids[indices]
            self._lats = all_lats[indices]
            self._lons = all_lons[indices]
            self._elevs = all_elevs[indices] if len(all_elevs) == total_points else np.zeros(self.max_points)
        else:
            self._selected_indices = None
            self._n_points = total_points
            self._station_ids = all_station_ids
            self._lats = all_lats
            self._lons = all_lons
            self._elevs = all_elevs if len(all_elevs) == total_points else np.zeros(total_points)

    def _get_point_axis(self, ds: xr.Dataset) -> int:
        """تشخیص محور نقاط در دیتاست (0 یا 1)"""
        if "point" in ds.dims:
            return list(ds.dims).index("point")
        return 1

    def load_block(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int,
        var_idx: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از داده‌های یک متغیر خاص.
        
        منطق:
            ۱. ابتدا در کش جستجو می‌کند (با اولویت block_size=1000, 2000, 5000)
            ۲. اگر کش پیدا شد، آن را برمی‌گرداند (و هیچ کش جدیدی نمی‌سازد)
            ۳. اگر کش پیدا نشد، از Zarr می‌خواند
            ۴. سپس کش جدید ذخیره می‌کند (فقط در صورتی که فایل کش وجود نداشته باشد)
        """
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None

        sample_hash = self._get_sample_hash()

        # ۱. جستجو در کش
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, var_idx, sample_hash)
            if cached is not None:
                return cached

        # ۲. خواندن از Zarr (چون کشی وجود نداشت)
        var_name = VARS[var_idx]
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        point_axis = self._get_point_axis(ds)

        arr = ds[var_name].values

        # اعمال نمونه‌برداری
        if self._selected_indices is not None:
            if point_axis == 0:
                arr = arr[self._selected_indices, :]
            else:
                arr = arr[:, self._selected_indices]

        # برش بلوک
        if point_axis == 0:
            arr_block = arr[block_start:block_start + block_size, :]
            arr_block = arr_block.T
        else:
            arr_block = arr[:, block_start:block_start + block_size]

        ds.close()

        # ۳. ذخیره در کش (فقط اگر فایل کش وجود نداشته باشد)
        if self.cache:
            self.cache.set(block_start, block_size, year_idx, month, var_idx, arr_block, sample_hash)

        return arr_block

    def load_block_all_vars(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از همه متغیرها به‌صورت همزمان.
        
        منطق:
            ۱. ابتدا در کش جستجو می‌کند (با کلید "all_vars")
            ۲. اگر کش پیدا شد، آن را برمی‌گرداند
            ۳. اگر کش پیدا نشد، از Zarr می‌خواند
            ۴. سپس کش جدید ذخیره می‌کند (فقط در صورتی که فایل کش وجود نداشته باشد)
        """
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None

        sample_hash = self._get_sample_hash()

        # ۱. جستجو در کش
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, "all_vars", sample_hash)
            if cached is not None:
                return cached

        # ۲. خواندن از Zarr (چون کشی وجود نداشت)
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        point_axis = self._get_point_axis(ds)
        n_vars = len(VARS)
        combined = None

        for v, var_name in enumerate(VARS):
            if var_name not in ds:
                continue
            arr = ds[var_name].values

            # اعمال نمونه‌برداری
            if self._selected_indices is not None:
                if point_axis == 0:
                    arr = arr[self._selected_indices, :]
                else:
                    arr = arr[:, self._selected_indices]

            # برش بلوک
            if point_axis == 0:
                arr_block = arr[block_start:block_start + block_size, :]
                arr_block = arr_block.T
            else:
                arr_block = arr[:, block_start:block_start + block_size]

            if combined is None:
                days = arr_block.shape[0]
                combined = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
            combined[:, :, v] = arr_block

        ds.close()

        # ۳. ذخیره در کش (فقط اگر فایل کش وجود نداشته باشد)
        if self.cache and combined is not None:
            self.cache.set(block_start, block_size, year_idx, month, "all_vars", combined, sample_hash)

        return combined

    def get_coords(self) -> Dict[str, Any]:
        """برگرداندن مختصات ایستگاه‌ها"""
        return {
            "stationid": self._station_ids,
            "lat": self._lats,
            "lon": self._lons,
            "elev": self._elevs,
            "point_dim": "point",
        }


class GriddedDataAdapter(BaseDataAdapter):
    """
    Adapter برای داده‌های شبکه‌ای (Gridded) با قابلیت محدوده‌ی مکانی (lat_min, lat_max, ...)
    """

    def __init__(
        self,
        zarr_base: str,
        year_list: List[int],
        cache_enabled: bool = True,
        max_points: int = 40000,
        lat_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_min: Optional[float] = None,
        lon_max: Optional[float] = None
    ):
        """
        پارامترها:
            zarr_base: مسیر پایه‌ی فایل‌های Zarr
            year_list: لیست سال‌ها
            cache_enabled: فعال/غیرفعال کردن کش
            max_points: حداکثر تعداد نقاط (برای نمونه‌برداری)
            lat_min, lat_max: محدوده‌ی عرض جغرافیایی
            lon_min, lon_max: محدوده‌ی طول جغرافیایی
        """
        super().__init__(zarr_base, year_list, cache_enabled)
        self.max_points = max_points
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        self._lat_indices: Optional[np.ndarray] = None
        self._lon_indices: Optional[np.ndarray] = None
        self._initialize_from_sample()

    def _initialize_from_sample(self) -> None:
        """
        بارگذاری شبکه و اعمال محدوده و نمونه‌برداری.
        """
        zarr_files = glob.glob(os.path.join(self.zarr_base, "*.zarr"))
        if not zarr_files:
            raise FileNotFoundError(f"No Zarr files found in {self.zarr_base}")

        ds = xr.open_zarr(zarr_files[0], consolidated=False)

        # استخراج طول و عرض
        if "latitude" in ds.coords:
            all_lats = ds["latitude"].values
        else:
            all_lats = ds["latitude"].values if "latitude" in ds.data_vars else None

        if "longitude" in ds.coords:
            all_lons = ds["longitude"].values
        else:
            all_lons = ds["longitude"].values if "longitude" in ds.data_vars else None

        if all_lats is None or all_lons is None:
            raise ValueError("Could not find latitude/longitude in Zarr file")

        ds.close()

        # اعمال محدوده مکانی
        if self.lat_min is not None and self.lat_max is not None:
            lat_mask = (all_lats >= self.lat_min) & (all_lats <= self.lat_max)
        else:
            lat_mask = np.ones(len(all_lats), dtype=bool)

        if self.lon_min is not None and self.lon_max is not None:
            lon_mask = (all_lons >= self.lon_min) & (all_lons <= self.lon_max)
        else:
            lon_mask = np.ones(len(all_lons), dtype=bool)

        self._lats = all_lats[lat_mask]
        self._lons = all_lons[lon_mask]
        n_lat = len(self._lats)
        n_lon = len(self._lons)
        total_points = n_lat * n_lon

        # نمونه‌برداری تصادفی (با seed ثابت)
        if total_points > self.max_points:
            np.random.seed(42)
            indices = np.random.choice(total_points, size=self.max_points, replace=False)
            indices = np.sort(indices)
            self._selected_indices = indices
            self._n_points = self.max_points

            lat_grid, lon_grid = np.meshgrid(self._lats, self._lons, indexing="ij")
            lat_flat = lat_grid.flatten()[indices]
            lon_flat = lon_grid.flatten()[indices]
            self._lats = lat_flat
            self._lons = lon_flat
        else:
            self._selected_indices = None
            self._n_points = total_points
            lat_grid, lon_grid = np.meshgrid(self._lats, self._lons, indexing="ij")
            self._lats = lat_grid.flatten()
            self._lons = lon_grid.flatten()

        self._lat_indices = np.where(lat_mask)[0]
        self._lon_indices = np.where(lon_mask)[0]

    def load_block(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int,
        var_idx: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از داده‌های یک متغیر خاص (شبکه‌ای).
        
        منطق مشابه StationDataAdapter است.
        """
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None

        sample_hash = self._get_sample_hash()

        # ۱. جستجو در کش
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, var_idx, sample_hash)
            if cached is not None:
                return cached

        # ۲. خواندن از Zarr
        var_name = VARS[var_idx]
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        arr = ds[var_name].values

        # اعمال محدوده مکانی (برش شبکه)
        if arr.ndim == 3:
            arr = arr[:, self._lat_indices, :][:, :, self._lon_indices]
            arr = arr.reshape(arr.shape[0], -1)   # (time, points)
        else:
            arr = arr.reshape(-1, arr.shape[-1])  # fallback
            arr = arr.T                          # (time, points)

        # اعمال نمونه‌برداری
        if self._selected_indices is not None:
            arr = arr[:, self._selected_indices]

        # برش بلوک
        arr_block = arr[:, block_start:block_start + block_size]
        ds.close()

        # ۳. ذخیره در کش (فقط اگر فایل کش وجود نداشته باشد)
        if self.cache:
            self.cache.set(block_start, block_size, year_idx, month, var_idx, arr_block, sample_hash)

        return arr_block

    def load_block_all_vars(
        self,
        block_start: int,
        block_size: int,
        year_idx: int,
        month: int
    ) -> Optional[np.ndarray]:
        """
        بارگذاری یک بلوک از همه متغیرها به‌صورت همزمان (شبکه‌ای).
        
        منطق مشابه StationDataAdapter است.
        """
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None

        sample_hash = self._get_sample_hash()

        # ۱. جستجو در کش
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, "all_vars", sample_hash)
            if cached is not None:
                return cached

        # ۲. خواندن از Zarr
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        n_vars = len(VARS)
        combined = None

        for v, var_name in enumerate(VARS):
            if var_name not in ds:
                continue
            arr = ds[var_name].values

            if arr.ndim == 3:
                arr = arr[:, self._lat_indices, :][:, :, self._lon_indices]
                arr = arr.reshape(arr.shape[0], -1)
            else:
                arr = arr.reshape(-1, arr.shape[-1])
                arr = arr.T

            if self._selected_indices is not None:
                arr = arr[:, self._selected_indices]

            arr_block = arr[:, block_start:block_start + block_size]

            if combined is None:
                days = arr_block.shape[0]
                combined = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
            combined[:, :, v] = arr_block

        ds.close()

        # ۳. ذخیره در کش (فقط اگر فایل کش وجود نداشته باشد)
        if self.cache and combined is not None:
            self.cache.set(block_start, block_size, year_idx, month, "all_vars", combined, sample_hash)

        return combined

    def get_coords(self) -> Dict[str, Any]:
        """برگرداندن مختصات نقاط شبکه (به‌صورت flattened)"""
        return {
            "lat": self._lats,
            "lon": self._lons,
            "elev": np.zeros(self._n_points),
            "point_dim": "point",
            "gridded": True,
        }


# ============================================================================
# ۴. تابع کارخانه (Factory) برای ساخت Adapter مناسب
# ============================================================================

def create_adapter(
    zarr_base: str,
    year_list: List[int],
    data_format: str = "auto",
    cache_enabled: bool = True,
    max_points: int = 40000,
    **kwargs
) -> BaseDataAdapter:
    """
    ساخت Adapter مناسب بر اساس فرمت داده (station, gridded, auto)
    
    پارامترها:
        zarr_base: مسیر پایه‌ی فایل‌های Zarr
        year_list: لیست سال‌ها
        data_format: 'station', 'gridded', یا 'auto'
        cache_enabled: فعال/غیرفعال کردن کش
        max_points: حداکثر تعداد نقاط
        **kwargs: پارامترهای اضافی مانند lat_min, lat_max, lon_min, lon_max
    
    بازگشت:
        نمونه‌ای از BaseDataAdapter
    """
    if data_format == "station":
        return StationDataAdapter(zarr_base, year_list, cache_enabled, max_points)
    
    elif data_format == "gridded":
        return GriddedDataAdapter(
            zarr_base,
            year_list,
            cache_enabled,
            max_points,
            lat_min=kwargs.get("lat_min"),
            lat_max=kwargs.get("lat_max"),
            lon_min=kwargs.get("lon_min"),
            lon_max=kwargs.get("lon_max")
        )
    
    elif data_format == "auto":
        zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
        if not zarr_files:
            raise FileNotFoundError(f"No Zarr files found in {zarr_base}")
        
        format_type = detect_data_format(zarr_files[0])
        
        if format_type == "station":
            return StationDataAdapter(zarr_base, year_list, cache_enabled, max_points)
        elif format_type == "gridded":
            return GriddedDataAdapter(
                zarr_base,
                year_list,
                cache_enabled,
                max_points,
                lat_min=kwargs.get("lat_min"),
                lat_max=kwargs.get("lat_max"),
                lon_min=kwargs.get("lon_min"),
                lon_max=kwargs.get("lon_max")
            )
        else:
            raise ValueError(f"Unsupported data format: {format_type}")
    
    else:
        raise ValueError(f"Unknown data_format: {data_format}")