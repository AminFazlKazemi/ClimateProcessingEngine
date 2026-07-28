#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_percentile_maps.py
================================================================================
تولید نقشه‌های صدک از خروجی Zarr بر اساس بهترین توزیع برازش‌شده.
================================================================================
"""

import os
import sys
import numpy as np
import xarray as xr
import zarr
from scipy import stats
from scipy.stats import norm, skewnorm, genextreme, pearson3
from tqdm import tqdm

# ============================================================
# تنظیمات
# ============================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"  # مسیر خروجی Zarr
OUTPUT_DIR = r"I:/climatology_366_rolling/percentile_maps"  # محل ذخیره نقشه‌ها
PERCENTILES = [0.9, 0.95, 0.99]  # صدک‌های مورد نظر
DAYS_TO_PLOT = [1, 91, 181, 271]  # روزهای خاص برای نقشه (اختیاری - اگر خالی باشد همه روزها)

# ============================================================
# توابع محاسبه ppf برای هر توزیع
# ============================================================
def ppf_normal(p, params):
    """ppf برای توزیع نرمال (p1=mean, p2=std)"""
    mu, sigma = params[0], params[1]
    if sigma <= 0:
        return np.nan
    return norm.ppf(p, loc=mu, scale=sigma)

def ppf_skew(p, params):
    """ppf برای توزیع skew-normal (p1=alpha, p2=loc, p3=scale)"""
    alpha, loc, scale = params[0], params[1], params[2]
    if scale <= 0:
        return np.nan
    try:
        return skewnorm.ppf(p, alpha, loc=loc, scale=scale)
    except:
        return np.nan

def ppf_gev(p, params):
    """ppf برای توزیع GEV (p1=shape, p2=loc, p3=scale)"""
    shape, loc, scale = params[0], params[1], params[2]
    if scale <= 0:
        return np.nan
    try:
        return genextreme.ppf(p, shape, loc=loc, scale=scale)
    except:
        return np.nan

def ppf_pearson(p, params):
    """ppf برای توزیع پیرسون نوع III (p1=shape, p2=scale, p3=loc)"""
    shape, scale, loc = params[0], params[1], params[2]
    if scale <= 0 or shape <= 0:
        return np.nan
    try:
        # pearson3 در scipy با پارامترهای (skew, loc, scale) تعریف می‌شود
        # اما ما shape, scale, loc داریم. باید تبدیل کنیم.
        # pearson3 از پارامتر skew استفاده می‌کند که با shape ارتباط دارد.
        # برای سادگی، از تابع pearson3 با همان پارامترها استفاده می‌کنیم.
        return pearson3.ppf(p, shape, loc=loc, scale=scale)
    except:
        return np.nan

def ppf_bimodal(p, params):
    """
    ppf برای توزیع دوگانه (مخلوط دو نرمال)
    params: (w1, mu1, sigma1, mu2, sigma2)
    """
    w1, mu1, sigma1, mu2, sigma2 = params
    if sigma1 <= 0 or sigma2 <= 0:
        return np.nan
    # برای محاسبه ppf دوگانه، از روش جستجوی دودویی روی CDF استفاده می‌کنیم
    # تابع CDF دوگانه:
    def cdf_bimodal(x):
        return w1 * norm.cdf(x, mu1, sigma1) + (1 - w1) * norm.cdf(x, mu2, sigma2)
    # جستجوی دودویی برای یافتن x به طوری که cdf(x) = p
    # محدوده جستجو: از min(mu1-5*sigma1, mu2-5*sigma2) تا max(mu1+5*sigma1, mu2+5*sigma2)
    low = min(mu1 - 5*sigma1, mu2 - 5*sigma2)
    high = max(mu1 + 5*sigma1, mu2 + 5*sigma2)
    # اگر محدوده خیلی کوچک است، گسترش دهید
    if high - low < 1e-6:
        low = mu1 - 10
        high = mu1 + 10
    for _ in range(100):  # ۱۰۰ تکرار کافی است
        mid = (low + high) / 2
        if cdf_bimodal(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2

# نگاشت کد توزیع به تابع ppf و تعداد پارامترها
PPF_FUNCS = {
    0: (ppf_normal, 2),      # normal
    1: (ppf_skew, 3),        # skew
    2: (ppf_gev, 3),         # gev
    3: (ppf_bimodal, 5),     # bimodal
    4: (ppf_pearson, 3),     # pearson
}


def get_params_for_distribution(var_name, dist_code, day_idx, ds):
    """
    استخراج پارامترهای توزیع برای یک متغیر، روز و کد توزیع مشخص
    """
    # نام‌های پایه برای پارامترها
    dist_name = {0: 'normal', 1: 'skew', 2: 'gev', 3: 'bimodal', 4: 'pearson'}[dist_code]
    n_params = PPF_FUNCS[dist_code][1]
    params = []
    for i in range(1, n_params + 1):
        key = f"{var_name}_{dist_name}_p{i}"
        if key in ds:
            params.append(ds[key].isel(day=day_idx).values)
        else:
            params.append(np.nan)
    return np.array(params)


def compute_percentile_map(zarr_path, percentiles, output_dir, days=None):
    """
    محاسبه و ذخیره نقشه‌های صدک برای روزهای مشخص

    Parameters
    ----------
    zarr_path : str
        مسیر فروشگاه Zarr خروجی
    percentiles : list of float
        لیست صدک‌های مورد نظر (مثلاً [0.9, 0.95, 0.99])
    output_dir : str
        مسیر ذخیره خروجی
    days : list of int, optional
        لیست روزهای سال (۰ تا ۳۶۵) برای محاسبه. اگر None باشد، همه روزها محاسبه می‌شوند.
    """
    os.makedirs(output_dir, exist_ok=True)

    # باز کردن فروشگاه Zarr
    print(f"📂 Reading Zarr store: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False)

    # استخراج مختصات
    n_days = ds.dims['day']
    n_points = ds.dims['point']
    lons = ds['lon'].values
    lats = ds['lat'].values

    # نام متغیرها (tmin, tmean, tmax)
    var_names = ['tmin', 'tmean', 'tmax']

    if days is None:
        days = list(range(n_days))

    print(f"📊 Processing {len(days)} days for {len(var_names)} variables and {len(percentiles)} percentiles...")

    # ایجاد لیست برای ذخیره نتایج (اختیاری - می‌توان مستقیماً فایل نوشت)
    for var_name in var_names:
        print(f"\n🔍 Processing variable: {var_name}")

        # پیدا کردن بهترین توزیع برای هر روز و هر نقطه
        best_dist_key = f"{var_name}_best_dist"
        if best_dist_key not in ds:
            print(f"⚠️ {best_dist_key} not found in dataset. Skipping {var_name}")
            continue

        best_dist = ds[best_dist_key].values  # shape: (day, point)

        for p in percentiles:
            print(f"   📈 Computing percentile {p*100:.0f}%...")
            # آرایه خروجی برای این صدک
            percentile_data = np.full((len(days), n_points), np.nan, dtype=np.float32)

            for idx, day_idx in enumerate(tqdm(days, desc=f"      Day loop")):
                for point_idx in range(n_points):
                    dist_code = best_dist[day_idx, point_idx]
                    if np.isnan(dist_code) or dist_code < 0:
                        continue
                    dist_code = int(dist_code)
                    # استخراج پارامترها
                    params = get_params_for_distribution(var_name, dist_code, day_idx, ds)
                    # اگر پارامترها شامل NaN هستند، ادامه نده
                    if np.any(np.isnan(params)):
                        continue
                    # محاسبه ppf
                    ppf_func = PPF_FUNCS[dist_code][0]
                    try:
                        val = ppf_func(p, params)
                        if not np.isnan(val) and not np.isinf(val):
                            percentile_data[idx, point_idx] = val
                    except:
                        continue

            # ذخیره به صورت xarray Dataset با مختصات
            ds_out = xr.Dataset(
                {
                    f"percentile_{int(p*100)}": (["day", "point"], percentile_data)
                },
                coords={
                    "day": days,
                    "point": np.arange(n_points),
                    "lon": ("point", lons),
                    "lat": ("point", lats)
                }
            )

            # ذخیره به صورت NetCDF
            nc_file = os.path.join(output_dir, f"{var_name}_percentile_{int(p*100)}.nc")
            ds_out.to_netcdf(nc_file)
            print(f"      💾 Saved NetCDF: {nc_file}")

            # (اختیاری) ذخیره به صورت GeoTIFF با استفاده از rioxarray
            try:
                import rioxarray
                # تبدیل به raster با ابعاد (day, lat, lon) با استفاده از rioxarray
                # ابتدا باید داده را به شکل (day, lat, lon) بازآرایی کنیم
                # برای این کار از xarray با مختصات dask استفاده می‌کنیم
                ds_raster = ds_out.rename({f"percentile_{int(p*100)}": "value"})
                # اضافه کردن مختصات lat/lon به عنوان ابعاد (نیاز به reshape)
                # روش ساده: ذخیره به صورت GeoTIFF برای هر روز جداگانه
                for day_idx in days:
                    day_data = ds_raster.isel(day=day_idx)
                    # ایجاد دیتاست با ابعاد (lat, lon)
                    day_ds = xr.Dataset(
                        {"value": (["lat", "lon"], day_data.value.values.reshape((len(lats), len(lons))))},
                        coords={"lat": lats, "lon": lons}
                    )
                    tif_file = os.path.join(output_dir, f"{var_name}_p{int(p*100)}_day{day_idx}.tif")
                    day_ds.rio.to_raster(tif_file)
                    print(f"         💾 Saved GeoTIFF: {tif_file}")
            except ImportError:
                print("      ℹ️ rioxarray not installed. Skipping GeoTIFF export.")

    print("\n✅ نقشه‌های صدک با موفقیت تولید شدند.")
    print(f"   📁 خروجی در: {output_dir}")


# ============================================================
# اجرای اصلی
# ============================================================
if __name__ == "__main__":
    compute_percentile_map(
        zarr_path=ZARR_PATH,
        percentiles=PERCENTILES,
        output_dir=OUTPUT_DIR,
        days=DAYS_TO_PLOT if DAYS_TO_PLOT else None
    )