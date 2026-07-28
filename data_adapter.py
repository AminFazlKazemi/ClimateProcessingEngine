#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_adapter.py
================================================================================
لایه انتزاع داده برای پشتیبانی از فرمتهای ایستگاهی و شبکهای
نسخه ۳.۳ - تشخیص ابعاد با استفاده از ds.dims
================================================================================
"""

import os
import glob
import numpy as np
import xarray as xr
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Optional, Tuple, List
from constants import VARS

# ============================================================================
# ۱. مدیریت کش (Disk Cache)
# ============================================================================

class DiskCache:
    """کش دیسکی برای ذخیره دادههای خوانده شده و کاهش I/O"""
    def __init__(self, cache_dir="./cache", max_size_gb=10, ttl_hours=24):
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_gb * 1024**3
        self.ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, block_start, block_size, year, month, var):
        key_str = f"{block_start}_{block_size}_{year}_{month}_{var}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, block_start, block_size, year, month, var):
        key = self._get_cache_key(block_start, block_size, year, month, var)
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if (datetime.now().timestamp() - mtime) < self.ttl_seconds:
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except:
                    pass
        return None

    def set(self, block_start, block_size, year, month, var, data):
        key = self._get_cache_key(block_start, block_size, year, month, var)
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) 
                         for f in os.listdir(self.cache_dir) 
                         if f.endswith(".pkl"))
        if total_size + data.nbytes > self.max_size_bytes:
            files = sorted([os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) 
                           if f.endswith(".pkl")], key=os.path.getmtime)
            while files and total_size + data.nbytes > self.max_size_bytes:
                os.remove(files.pop(0))
                total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) 
                                 for f in os.listdir(self.cache_dir) 
                                 if f.endswith(".pkl"))
        with open(path, "wb") as f:
            pickle.dump(data, f)

# ============================================================================
# ۲. شناسایی خودکار فرمت داده (اصلاح‌شده)
# ============================================================================

def detect_data_format(zarr_path):
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

def get_grid_info(zarr_path):
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

def get_station_info(zarr_path):
    ds = xr.open_zarr(zarr_path, consolidated=False)
    if "stationid" in ds.coords:
        station_ids = ds["stationid"].values
    else:
        station_ids = ds["stationid"].values if "stationid" in ds.data_vars else None
    if "lat" in ds.coords:
        lats = ds["lat"].values
    else:
        lats = ds["lat"].values if "lat" in ds.data_vars else None
    if "lon" in ds.coords:
        lons = ds["lon"].values
    else:
        lons = ds["lon"].values if "lon" in ds.data_vars else None
    if "elev" in ds.coords:
        elevs = ds["elev"].values
    elif "elev" in ds.data_vars:
        elevs = ds["elev"].values
    else:
        elevs = np.zeros_like(station_ids)
    ds.close()
    return station_ids, lats, lons, elevs

# ============================================================================
# ۳. کلاسهای Adapter
# ============================================================================

class BaseDataAdapter:
    def __init__(self, zarr_base, year_list, cache_enabled=True):
        self.zarr_base = zarr_base
        self.year_list = year_list
        self._file_map = None
        self._n_points = None
        self.cache = DiskCache() if cache_enabled else None

    @property
    def n_points(self):
        return self._n_points

    @property
    def file_map(self):
        if self._file_map is None:
            self._build_file_map()
        return self._file_map

    def _build_file_map(self):
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

    def load_block(self, block_start, block_size, year_idx, month, var_idx):
        raise NotImplementedError

    def load_block_all_vars(self, block_start, block_size, year_idx, month):
        raise NotImplementedError

    def get_coords(self):
        raise NotImplementedError


class StationDataAdapter(BaseDataAdapter):
    def __init__(self, zarr_base, year_list, cache_enabled=True, max_points=40000):
        super().__init__(zarr_base, year_list, cache_enabled)
        self.max_points = max_points
        first_file = None
        zarr_files = glob.glob(os.path.join(self.zarr_base, "*.zarr"))
        if zarr_files:
            first_file = zarr_files[0]
        if first_file:
            ds = xr.open_zarr(first_file, consolidated=False)
            total_points = ds.sizes["point"]

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

            if total_points > max_points:
                np.random.seed(42)
                indices = np.random.choice(total_points, size=max_points, replace=False)
                indices = np.sort(indices)
                self._selected_indices = indices
                self._n_points = max_points
                self._station_ids = all_station_ids[indices]
                self._lats = all_lats[indices]
                self._lons = all_lons[indices]
                self._elevs = all_elevs[indices] if len(all_elevs) == total_points else np.zeros(max_points)
            else:
                self._selected_indices = None
                self._n_points = total_points
                self._station_ids = all_station_ids
                self._lats = all_lats
                self._lons = all_lons
                self._elevs = all_elevs if len(all_elevs) == total_points else np.zeros(total_points)

            ds.close()

    def _get_point_axis(self, ds):
        """دریافت ایندکس محور نقاط از روی dims"""
        if "point" in ds.dims:
            return list(ds.dims).index("point")
        # fallback: بعد دوم را به عنوان نقطه فرض کن
        return 1

    def load_block(self, block_start, block_size, year_idx, month, var_idx):
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, var_idx)
            if cached is not None:
                return cached

        var_name = VARS[var_idx]
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        point_axis = self._get_point_axis(ds)

        arr = ds[var_name].values  # shape: (day, point) معمولاً

        # اعمال انتخاب نقاط نمونه‌گیری‌شده
        if self._selected_indices is not None:
            if point_axis == 0:
                arr = arr[self._selected_indices, :]
            else:
                arr = arr[:, self._selected_indices]

        # استخراج بلوک
        if point_axis == 0:
            arr_block = arr[block_start:block_start + block_size, :]  # (block_size, day)
            arr_block = arr_block.T  # (day, block_size)
        else:
            arr_block = arr[:, block_start:block_start + block_size]  # (day, block_size)

        ds.close()
        if self.cache:
            self.cache.set(block_start, block_size, year_idx, month, var_idx, arr_block)
        return arr_block

    def load_block_all_vars(self, block_start, block_size, year_idx, month):
        """بارگذاری همه متغیرها در یک بار باز کردن فایل (نسخه سریع)"""
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None

        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, "all_vars")
            if cached is not None:
                return cached

        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        point_axis = self._get_point_axis(ds)
        n_vars = len(VARS)
        combined = None

        for v, var_name in enumerate(VARS):
            if var_name not in ds:
                continue
            arr = ds[var_name].values

            # اعمال انتخاب نقاط نمونه‌گیری‌شده
            if self._selected_indices is not None:
                if point_axis == 0:
                    arr = arr[self._selected_indices, :]
                else:
                    arr = arr[:, self._selected_indices]

            # استخراج بلوک
            if point_axis == 0:
                arr_block = arr[block_start:block_start + block_size, :]  # (block_size, day)
                arr_block = arr_block.T  # (day, block_size)
            else:
                arr_block = arr[:, block_start:block_start + block_size]  # (day, block_size)

            if combined is None:
                days = arr_block.shape[0]
                combined = np.full((days, block_size, n_vars), np.nan, dtype=np.float32)
            combined[:, :, v] = arr_block

        ds.close()
        if self.cache and combined is not None:
            self.cache.set(block_start, block_size, year_idx, month, "all_vars", combined)
        return combined

    def get_coords(self):
        return {
            "stationid": self._station_ids,
            "lat": self._lats,
            "lon": self._lons,
            "elev": self._elevs,
            "point_dim": "point",
        }


class GriddedDataAdapter(BaseDataAdapter):
    def __init__(self, zarr_base, year_list, cache_enabled=True, max_points=40000,
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        super().__init__(zarr_base, year_list, cache_enabled)
        self.max_points = max_points
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

        first_file = None
        zarr_files = glob.glob(os.path.join(self.zarr_base, "*.zarr"))
        if zarr_files:
            first_file = zarr_files[0]
        if first_file:
            ds = xr.open_zarr(first_file, consolidated=False)

            if "latitude" in ds.coords:
                all_lats = ds["latitude"].values
            else:
                all_lats = ds["latitude"].values if "latitude" in ds.data_vars else None

            if "longitude" in ds.coords:
                all_lons = ds["longitude"].values
            else:
                all_lons = ds["longitude"].values if "longitude" in ds.data_vars else None

            if all_lats is None or all_lons is None:
                raise ValueError("Could not find latitude/longitude")

            if lat_min is not None and lat_max is not None:
                lat_mask = (all_lats >= lat_min) & (all_lats <= lat_max)
            else:
                lat_mask = np.ones(len(all_lats), dtype=bool)
            if lon_min is not None and lon_max is not None:
                lon_mask = (all_lons >= lon_min) & (all_lons <= lon_max)
            else:
                lon_mask = np.ones(len(all_lons), dtype=bool)

            self._lats = all_lats[lat_mask]
            self._lons = all_lons[lon_mask]
            n_lat = len(self._lats)
            n_lon = len(self._lons)
            total_points = n_lat * n_lon

            if total_points > max_points:
                np.random.seed(42)
                indices = np.random.choice(total_points, size=max_points, replace=False)
                indices = np.sort(indices)
                self._selected_indices = indices
                self._n_points = max_points
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
            ds.close()

    def load_block(self, block_start, block_size, year_idx, month, var_idx):
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, var_idx)
            if cached is not None:
                return cached

        var_name = VARS[var_idx]
        ds = xr.open_zarr(self.file_map[key], consolidated=False)
        arr = ds[var_name].values

        # تبدیل به (day, n_points)
        if arr.ndim == 3:
            arr = arr[:, self._lat_indices, :][:, :, self._lon_indices]
            arr = arr.reshape(arr.shape[0], -1)
        else:
            # fallback
            arr = arr.reshape(-1, arr.shape[-1])
            arr = arr.T

        if self._selected_indices is not None:
            arr = arr[:, self._selected_indices]

        arr_block = arr[:, block_start:block_start + block_size]
        ds.close()
        if self.cache:
            self.cache.set(block_start, block_size, year_idx, month, var_idx, arr_block)
        return arr_block

    def load_block_all_vars(self, block_start, block_size, year_idx, month):
        key = (self.year_list[year_idx], month)
        if key not in self.file_map:
            return None
        if self.cache:
            cached = self.cache.get(block_start, block_size, year_idx, month, "all_vars")
            if cached is not None:
                return cached

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
        if self.cache and combined is not None:
            self.cache.set(block_start, block_size, year_idx, month, "all_vars", combined)
        return combined

    def get_coords(self):
        return {
            "lat": self._lats,
            "lon": self._lons,
            "elev": np.zeros(self._n_points),
            "point_dim": "point",
            "gridded": True,
        }


def create_adapter(zarr_base, year_list, data_format="auto", cache_enabled=True,
                   max_points=40000, **kwargs):
    if data_format == "station":
        return StationDataAdapter(zarr_base, year_list, cache_enabled, max_points)
    elif data_format == "gridded":
        return GriddedDataAdapter(zarr_base, year_list, cache_enabled, max_points, **kwargs)
    elif data_format == "auto":
        zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
        if not zarr_files:
            raise FileNotFoundError(f"No Zarr files found in {zarr_base}")
        format_type = detect_data_format(zarr_files[0])
        if format_type == "station":
            return StationDataAdapter(zarr_base, year_list, cache_enabled, max_points)
        elif format_type == "gridded":
            return GriddedDataAdapter(zarr_base, year_list, cache_enabled, max_points, **kwargs)
        else:
            raise ValueError(f"Unsupported data format: {format_type}")
    else:
        raise ValueError(f"Unknown data_format: {data_format}")