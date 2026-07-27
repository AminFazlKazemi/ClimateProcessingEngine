#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climatology Processing Engine v3.0 - مستقل و بدون وابستگی به کش
محاسبه اقلیم‌شناسی روزانه (میانگین‌های متحرک) برای دوره 1369-1399
خروجی: Zarr با ابعاد (day_of_year, station)
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.dataframe as dd
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta

# ============================================================
# تنظیمات اولیه
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# پارامترها (مشابه کد اصلی شما)
# ============================================================
START_YEAR = 1369
END_YEAR = 1399
TOTAL_YEARS = END_YEAR - START_YEAR + 1  # 31
DAYS_IN_YEAR = 366  # سال کبیسه
OUTPUT_PATH = "I:/climatology_366_rolling/climatology_stationwise_final.zarr"
N_OUTPUTS = 31
MAX_VALUES = 155

# ============================================================
# ۱. ساخت جدول تقویم (مشابه calendar.txt)
# ============================================================
def build_calendar_table():
    """ساخت جدول روزهای سال برای تمام سال‌های دوره"""
    logger.info("Building calendar table...")
    
    # تولید تاریخ‌های شمسی (با استفاده از کتابخانه jdatetime)
    try:
        import jdatetime
    except ImportError:
        logger.error("jdatetime not installed. Install with: pip install jdatetime")
        raise
    
    dates = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            days_in_month = 31 if month <= 6 else (30 if month < 12 else (29 if jdatetime.date(year, 12, 1).is_leap() else 30))
            for day in range(1, days_in_month + 1):
                dates.append(jdatetime.date(year, month, day))
    
    # تبدیل به DataFrame
    df = pd.DataFrame({
        'year': [d.year for d in dates],
        'month': [d.month for d in dates],
        'day': [d.day for d in dates],
        'doy': [(d - jdatetime.date(START_YEAR, 1, 1)).days + 1 for d in dates]  # روز از ابتدای دوره
    })
    
    # مرتب‌سازی بر اساس سال و روز
    df = df.sort_values(['year', 'doy']).reset_index(drop=True)
    
    # ساخت آرایه سه‌بعدی (year, month, day)
    years = sorted(df['year'].unique())
    months = sorted(df['month'].unique())
    days = sorted(df['day'].unique())
    
    # پوسته (dummy) برای ذخیره جدول
    doy_table = np.zeros((len(years), len(months), len(days)), dtype=np.int32)
    for i, y in enumerate(years):
        for j, m in enumerate(months):
            for k, d in enumerate(days):
                mask = (df['year'] == y) & (df['month'] == m) & (df['day'] == d)
                if mask.any():
                    doy_table[i, j, k] = df.loc[mask, 'doy'].iloc[0]
                else:
                    doy_table[i, j, k] = -1  # مقدار پیش‌فرض
    
    logger.info(f"   ✅ doy_table shape: {doy_table.shape}")
    return doy_table, df

# ============================================================
# ۲. خواندن داده‌های ایستگاهی
# ============================================================
def read_station_data(data_dir: str):
    """خواندن داده‌های روزانه ایستگاه‌ها از فایل‌های NetCDF یا CSV"""
    logger.info("Reading station data...")
    
    # مثال: خواندن از فایل‌های NetCDF (فرض بر این است که هر سال یک فایل دارد)
    # در اینجا باید مسیر واقعی داده‌های شما قرار گیرد
    # برای نمونه، یک دیتاست ساختگی می‌سازیم
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} not found. Generating synthetic data for testing...")
        return generate_synthetic_data()
    
    # خواندن فایل‌های NetCDF (در صورت وجود)
    files = list(data_path.glob("*.nc"))
    if not files:
        logger.warning("No NetCDF files found. Generating synthetic data...")
        return generate_synthetic_data()
    
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True)
    return ds

def generate_synthetic_data():
    """تولید داده‌های ساختگی برای تست (بدون داده واقعی)"""
    logger.info("Generating synthetic station data for testing...")
    
    n_stations = 1000  # تعداد ایستگاه‌های ساختگی
    n_days = TOTAL_YEARS * DAYS_IN_YEAR
    years = np.repeat(np.arange(START_YEAR, END_YEAR + 1), DAYS_IN_YEAR)
    doys = np.tile(np.arange(1, DAYS_IN_YEAR + 1), TOTAL_YEARS)
    
    # تولید مقادیر تصادفی با روند فصلی
    base_temp = 15 + 10 * np.sin(2 * np.pi * doys / DAYS_IN_YEAR) + np.random.randn(n_days) * 2
    station_offsets = np.random.randn(n_stations) * 3
    
    # ساخت دیتاست xarray
    coords = {
        'station': np.arange(n_stations),
        'time': pd.date_range(
            start=f'{START_YEAR}-01-01',
            periods=n_days,
            freq='D'
        )
    }
    
    data = base_temp[:, None] + station_offsets[None, :] + np.random.randn(n_days, n_stations) * 0.5
    ds = xr.Dataset(
        {
            'tas': (['time', 'station'], data),
            'year': ('time', years),
            'doy': ('time', doys)
        },
        coords=coords
    )
    
    logger.info(f"   ✅ Synthetic dataset shape: {ds.tas.shape}")
    return ds

# ============================================================
# ۳. محاسبه اقلیم‌شناسی (میانگین‌های متحرک)
# ============================================================
def compute_climatology(ds, doy_table, window_sizes=[31, 61, 91, 121, 151]):
    """محاسبه آماره‌های اقلیمی برای پنجره‌های مختلف"""
    logger.info("Computing climatology with rolling windows...")
    
    # گروه‌بندی بر اساس روز سال (doy) و ایستگاه
    climatology_results = {}
    
    for window in window_sizes:
        logger.info(f"   Processing window size: {window}")
        
        # میانگین متحرک روی بعد زمان
        rolling_mean = ds.tas.rolling(time=window, center=True).mean()
        
        # گروه‌بندی بر اساس doy (روز سال) و ایستگاه
        climatology = rolling_mean.groupby('doy').mean(dim='time')
        
        # ذخیره در دیکشنری
        climatology_results[f'window_{window}'] = climatology
    
    return climatology_results

# ============================================================
# ۴. ذخیره خروجی به Zarr
# ============================================================
def save_to_zarr(climatology_results, output_path):
    """ذخیره نتایج به‌صورت Zarr"""
    logger.info(f"Saving results to Zarr: {output_path}")
    
    # ترکیب همه پنجره‌ها در یک دیتاست
    combined = xr.Dataset()
    for name, ds in climatology_results.items():
        combined[name] = ds
    
    # ذخیره به Zarr
    combined.to_zarr(output_path, mode='w', consolidated=True)
    logger.info(f"   ✅ Saved to {output_path}")

# ============================================================
# ۵. تابع اصلی
# ============================================================
def main():
    """اجرای کل فرایند"""
    logger.info("=" * 70)
    logger.info("🚀 CLIMATOLOGY PROCESSING ENGINE v3.0 (Independent)")
    logger.info(f"   Years: {START_YEAR}–{END_YEAR} ({TOTAL_YEARS} years)")
    logger.info(f"   Days: {DAYS_IN_YEAR}")
    logger.info(f"   Output: {OUTPUT_PATH}")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    try:
        # مرحله ۱: جدول تقویم
        doy_table, df_calendar = build_calendar_table()
        
        # مرحله ۲: خواندن داده‌ها
        data_dir = "K:/gozareshha/dr vazife/140504 - qc temp/climatology/data"  # مسیر واقعی را تنظیم کنید
        ds = read_station_data(data_dir)
        
        # مرحله ۳: محاسبه اقلیم‌شناسی
        window_sizes = [31, 61, 91, 121, 151]  # پنجره‌های 31, 61, 91, 121, 151 روزه
        climatology_results = compute_climatology(ds, doy_table, window_sizes)
        
        # مرحله ۴: ذخیره خروجی
        save_to_zarr(climatology_results, OUTPUT_PATH)
        
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"✅ PROCESSING COMPLETED in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        logger.info(f"   Output saved to: {OUTPUT_PATH}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

# ============================================================
# اجرای اصلی
# ============================================================
if __name__ == "__main__":
    main()