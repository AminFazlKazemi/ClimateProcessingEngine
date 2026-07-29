#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_percentile_maps.py - نسخه بهینه‌شده با Dask و محاسبات برداری
================================================================================
تولید نقشه‌های صدک از خروجی Zarr بر اساس بهترین توزیع برازش‌شده.
استفاده از Dask برای پردازش موازی و کاهش چشمگیر زمان.
================================================================================
"""

import os
import numpy as np
import xarray as xr
import dask.array as da
from scipy.stats import norm, skewnorm, genextreme, pearson3
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# تنظیمات
# ============================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
OUTPUT_DIR = r"I:/climatology_366_rolling/percentile_maps"
PERCENTILES = [0.9, 0.95, 0.99]
DAYS_TO_PLOT = [1, 91, 181, 271]  # روزهای خاص (۱-بیس)

# ============================================================
# توابع ppf برداری برای هر توزیع
# ============================================================
def ppf_normal(p, params):
    """ppf برای توزیع نرمال (p1=mean, p2=std) - کاملاً برداری"""
    mu, sigma = params[..., 0], params[..., 1]
    with np.errstate(invalid='ignore'):
        return np.where(sigma > 0, norm.ppf(p, loc=mu, scale=sigma), np.nan)

def ppf_skew(p, params):
    """ppf برای skew-normal (p1=alpha, p2=loc, p3=scale) - برداری"""
    alpha, loc, scale = params[..., 0], params[..., 1], params[..., 2]
    with np.errstate(invalid='ignore'):
        return np.where(scale > 0, skewnorm.ppf(p, alpha, loc=loc, scale=scale), np.nan)

def ppf_gev(p, params):
    """ppf برای GEV (p1=shape, p2=loc, p3=scale) - برداری"""
    shape, loc, scale = params[..., 0], params[..., 1], params[..., 2]
    with np.errstate(invalid='ignore'):
        return np.where(scale > 0, genextreme.ppf(p, shape, loc=loc, scale=scale), np.nan)

def ppf_pearson(p, params):
    """ppf برای پیرسون نوع III (p1=shape, p2=scale, p3=loc) - برداری"""
    shape, scale, loc = params[..., 0], params[..., 1], params[..., 2]
    with np.errstate(invalid='ignore'):
        return np.where((scale > 0) & (shape > 0), pearson3.ppf(p, shape, loc=loc, scale=scale), np.nan)

def ppf_bimodal_vectorized(p, params):
    """
    ppf برای توزیع دوگانه (مخلوط دو نرمال) - با استفاده از جستجوی دودویی برداری
    params: (w1, mu1, sigma1, mu2, sigma2)
    """
    w1, mu1, sigma1, mu2, sigma2 = params[..., 0], params[..., 1], params[..., 2], params[..., 3], params[..., 4]
    
    # تابع CDF دوگانه
    def cdf_bimodal(x, w1, mu1, sigma1, mu2, sigma2):
        return w1 * norm.cdf(x, mu1, sigma1) + (1 - w1) * norm.cdf(x, mu2, sigma2)
    
    # جستجوی دودویی با استفاده از جستجوی برداری (برای هر نقطه به صورت مجزا)
    # این بخش همچنان نیاز به حلقه دارد، اما تعداد نقاط با توزیع دوگانه معمولاً کم است.
    # برای تعداد زیاد، می‌توان از روش تقریبی یا نمونه‌برداری استفاده کرد.
    # در اینجا از یک حلقه ساده استفاده می‌کنیم که برای تعداد کم قابل قبول است.
    low = np.minimum(mu1 - 5*sigma1, mu2 - 5*sigma2)
    high = np.maximum(mu1 + 5*sigma1, mu2 + 5*sigma2)
    # برای مواردی که محدوده خیلی کوچک است، گسترش دهید
    low = np.where(high - low < 1e-6, mu1 - 10, low)
    high = np.where(high - low < 1e-6, mu1 + 10, high)
    
    # مقداردهی اولیه
    x = (low + high) / 2
    for _ in range(80):  # ۸۰ تکرار کافی است
        cdf_val = cdf_bimodal(x, w1, mu1, sigma1, mu2, sigma2)
        low = np.where(cdf_val < p, x, low)
        high = np.where(cdf_val >= p, x, high)
        x = (low + high) / 2
    return x

# نگاشت کد توزیع به تابع ppf و تعداد پارامترها
PPF_FUNCS = {
    0: (ppf_normal, 2),
    1: (ppf_skew, 3),
    2: (ppf_gev, 3),
    3: (ppf_bimodal_vectorized, 5),
    4: (ppf_pearson, 3),
}

# ============================================================
# تابع اصلی با استفاده از Dask و xarray
# ============================================================
def compute_percentile_maps_fast(zarr_path, percentiles, output_dir, days=None):
    """
    محاسبه صدک‌ها با استفاده از Dask و xarray.apply_ufunc
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # باز کردن Zarr با chunks مناسب برای Dask
    print(f"📂 Reading Zarr store: {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False, chunks={'day': 10, 'point': 5000})
    
    # استخراج مختصات
    n_days = ds.dims['day']
    n_points = ds.dims['point']
    lons = ds['lon'].values
    lats = ds['lat'].values
    
    var_names = ['tmin', 'tmean', 'tmax']
    if days is None:
        days = list(range(n_days))
    
    print(f"📊 Processing {len(days)} days for {len(var_names)} variables and {len(percentiles)} percentiles...")
    
    # نام توزیع‌ها برای دسترسی به پارامترها
    dist_names = {0: 'normal', 1: 'skew', 2: 'gev', 3: 'bimodal', 4: 'pearson'}
    
    for var_name in var_names:
        print(f"\n🔍 Processing variable: {var_name}")
        
        best_dist_key = f"{var_name}_best_dist"
        if best_dist_key not in ds:
            print(f"⚠️ {best_dist_key} not found. Skipping {var_name}")
            continue
        
        # بارگذاری best_dist به عنوان dask array
        best_dist = ds[best_dist_key].values  # این هنوز dask array است
        
        # برای هر صدک
        for p in percentiles:
            print(f"   📈 Computing percentile {p*100:.0f}%...")
            
            # ایجاد آرایه خالی برای نتایج (با استفاده از dask)
            percentile_data = np.full((len(days), n_points), np.nan, dtype=np.float32)
            
            # برای هر روز به صورت جداگانه محاسبه می‌کنیم (چون ppf به روز وابسته است)
            for idx, day_idx in enumerate(tqdm(days, desc=f"      Days for {var_name} p{p*100:.0f}")):
                # استخراج کد توزیع برای این روز
                dist_codes = best_dist[day_idx, :].compute()  # تبدیل به numpy برای این روز
                
                # ایجاد آرایه پارامترها برای این روز (به صورت دیکشنری)
                params_dict = {}
                for dist_code, (_, n_params) in PPF_FUNCS.items():
                    dist_name = dist_names[dist_code]
                    params_list = []
                    for i in range(1, n_params + 1):
                        key = f"{var_name}_{dist_name}_p{i}"
                        if key in ds:
                            # استخراج داده برای این روز و تبدیل به numpy
                            params_list.append(ds[key].isel(day=day_idx).values.compute())
                        else:
                            params_list.append(np.full(n_points, np.nan))
                    params_dict[dist_code] = np.stack(params_list, axis=-1)  # shape: (n_points, n_params)
                
                # محاسبه ppf برای هر نقطه بر اساس کد توزیع
                result = np.full(n_points, np.nan, dtype=np.float32)
                
                # برای هر توزیع، نقاط مربوطه را پیدا کرده و محاسبه کن
                for dist_code, (ppf_func, n_params) in PPF_FUNCS.items():
                    mask = (dist_codes == dist_code)
                    if not np.any(mask):
                        continue
                    params = params_dict[dist_code][mask]  # shape: (n_points_masked, n_params)
                    # محاسبه ppf برای این نقاط
                    try:
                        vals = ppf_func(p, params)
                        result[mask] = vals
                    except Exception as e:
                        print(f"      ⚠️ Error in ppf for dist {dist_code}: {e}")
                        continue
                
                # ذخیره نتیجه برای این روز
                percentile_data[idx, :] = result
            
            # ذخیره نتایج به صورت NetCDF
            ds_out = xr.Dataset(
                {f"percentile_{int(p*100)}": (["day", "point"], percentile_data)},
                coords={
                    "day": days,
                    "point": np.arange(n_points),
                    "lon": ("point", lons),
                    "lat": ("point", lats)
                }
            )
            nc_file = os.path.join(output_dir, f"{var_name}_percentile_{int(p*100)}.nc")
            ds_out.to_netcdf(nc_file)
            print(f"      💾 Saved NetCDF: {nc_file}")
            
            # (اختیاری) ذخیره به صورت GeoTIFF برای هر روز
            try:
                import rioxarray
                for day_idx in days:
                    day_data = ds_out.isel(day=day_idx)
                    # بازآرایی به (lat, lon)
                    day_reshaped = day_data[f"percentile_{int(p*100)}"].values.reshape((len(lats), len(lons)))
                    day_ds = xr.Dataset(
                        {"value": (["lat", "lon"], day_reshaped)},
                        coords={"lat": lats, "lon": lons}
                    )
                    tif_file = os.path.join(output_dir, f"{var_name}_p{int(p*100)}_day{day_idx}.tif")
                    day_ds.rio.to_raster(tif_file)
                print(f"      💾 Saved GeoTIFFs for each day.")
            except ImportError:
                pass  # rioxarray not installed
    
    print("\n✅ نقشه‌های صدک با موفقیت تولید شدند.")
    print(f"   📁 خروجی در: {output_dir}")


# ============================================================
# اجرای اصلی
# ============================================================
if __name__ == "__main__":
    compute_percentile_maps_fast(
        zarr_path=ZARR_PATH,
        percentiles=PERCENTILES,
        output_dir=OUTPUT_DIR,
        days=DAYS_TO_PLOT if DAYS_TO_PLOT else None
    )