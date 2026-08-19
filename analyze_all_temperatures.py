#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_all_temperatures.py
================================================================================
تحلیل جامع و مقایسه‌ای توزیع‌های برازش‌شده برای هر سه متغیر دمایی (Tmax, Tmean, Tmin)
نسخه نهایی – تمام امکانات در یک اسکریپت
================================================================================
- خواندن فایل Zarr اقلیم‌شناسی (خروجی نهایی پروژه)
- استفاده از ۵ توزیع: Normal, Skew‑Normal, GEV, Bimodal, Pearson
- تحلیل روزانه، فصلی، ارتفاعی و مکانی برای هر متغیر
- نمایش درصد میانگین سالانه هر توزیع در نقشه‌ها
- مقایسه‌ی مستقیم سه متغیر در نمودارهای مشترک
- خروجی‌های کامل شامل جداول و تصاویر با پسوند مشخص
- نمایش صحیح متون فارسی با bidi + arabic_reshaper
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bidi.algorithm import get_display
from arabic_reshaper import reshape
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# تنظیمات فونت و استایل
# ============================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ============================================================================
# توابع کمکی برای نمایش فارسی
# ============================================================================
def persian_text(text):
    """تبدیل متن فارسی به شکل قابل نمایش در matplotlib"""
    reshaped = reshape(text)
    bidi_text = get_display(reshaped)
    return bidi_text

def en_to_fa(num, formatter='%1.1f%%'):
    """تبدیل عدد به رشته فارسی با فرمت دلخواه"""
    num_as_string = formatter % num
    mapping = dict(zip('0123456789.%', '۰۱۲۳۴۵۶۷۸۹.%'))
    return ''.join(mapping[digit] for digit in num_as_string)

# ============================================================================
# ۰. تعریف متغیرهای هدف (لیست تمام متغیرهای دمایی)
# ============================================================================
VARS = ['tmax', 'tmean', 'tmin']   # می‌توانید هر کدام را حذف یا اضافه کنید
VAR_LABELS = {
    'tmax': 'دمای بیشینه',
    'tmean': 'دمای میانگین',
    'tmin': 'دمای کمینه'
}
VAR_COLORS = {
    'tmax': '#e41a1c',
    'tmean': '#377eb8',
    'tmin': '#4daf4a'
}   # برای نمودارهای مقایسه‌ای

# ============================================================================
# ۱. تنظیمات و بارگذاری داده‌ها
# ============================================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
OUTPUT_DIR = os.path.dirname(ZARR_PATH) + "/analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("📊 تحلیل جامع و مقایسه‌ای توزیع‌های دمایی")
print("=" * 80)

if not os.path.exists(ZARR_PATH):
    print(f"❌ فایل Zarr یافت نشد: {ZARR_PATH}")
    sys.exit()

ds = xr.open_zarr(ZARR_PATH, consolidated=False)
print(f"✅ فایل بارگذاری شد. ابعاد: {list(ds.dims)}")

# تشخیص ابعاد
day_dim = None
point_dim = None
for d in list(ds.dims):
    if ds.dims[d] in [365, 366]:
        day_dim = d
    else:
        point_dim = d

if day_dim is None or point_dim is None:
    raise ValueError("ابعاد روز و نقطه شناسایی نشد.")

n_days = ds.dims[day_dim]
n_points = ds.dims[point_dim]
print(f"   بعد روز: {day_dim} ({n_days})")
print(f"   بعد نقطه: {point_dim} ({n_points})")

# مختصات ایستگاه‌ها
elev = ds['elev'].values
if elev.ndim > 1:
    elev = elev.flatten()
lat = ds['lat'].values
if lat.ndim > 1:
    lat = lat.flatten()
lon = ds['lon'].values
if lon.ndim > 1:
    lon = lon.flatten()

# ثابت‌های کلی
dist_codes = [0, 1, 2, 3, 4]
dist_labels = ['Normal', 'Skew-Normal', 'GEV', 'Bimodal', 'Pearson']
dist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ============================================================================
# ۲. توابع کمکی مشترک
# ============================================================================
def get_season(day_idx):
    day = day_idx + 1
    if day <= 90:
        return 'زمستان'
    elif day <= 181:
        return 'بهار'
    elif day <= 273:
        return 'تابستان'
    elif day <= 365:
        return 'پاییز'
    else:
        return 'زمستان'

def to_1d(arr, length, default_val=np.nan):
    if arr is None:
        return np.full(length, default_val)
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    if arr.size == 1:
        return np.full(length, arr[0])
    if arr.size != length:
        if arr.size < length:
            new_arr = np.full(length, default_val)
            new_arr[:arr.size] = arr
            return new_arr
        else:
            return arr[:length]
    return arr

def reshape_array(arr, n_days, n_points):
    """تبدیل آرایه به شکل (day, point) در صورت نیاز"""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size == n_points:
            return np.tile(arr, (n_days, 1))
        elif arr.size == n_days:
            return np.tile(arr, (n_points, 1)).T
        else:
            return arr.reshape(n_days, n_points) if arr.size == n_days*n_points else None
    elif arr.ndim == 2:
        if arr.shape[0] == n_days and arr.shape[1] == n_points:
            return arr
        elif arr.shape[0] == n_points and arr.shape[1] == n_days:
            return arr.T
        else:
            return arr.reshape(n_days, n_points) if arr.size == n_days*n_points else None
    else:
        return None

# ============================================================================
# ۳. تابع تحلیل یک متغیر (برگرفته از کد قبلی با بهبود)
# ============================================================================
def analyze_single_variable(var_name, ds, n_days, n_points, day_dim, point_dim,
                            elev, lat, lon, output_dir):
    print(f"\n{'='*60}")
    print(f"🔹 تحلیل متغیر: {var_name} ({VAR_LABELS.get(var_name, var_name)})")
    print('='*60)

    # استخراج داده‌های اصلی
    best_dist = ds[f'{var_name}_best_dist'].values
    if best_dist.ndim == 2:
        if best_dist.shape[0] != n_days or best_dist.shape[1] != n_points:
            if best_dist.shape[0] == n_points and best_dist.shape[1] == n_days:
                best_dist = best_dist.T
            else:
                best_dist = best_dist.reshape(n_days, n_points)
    else:
        raise ValueError("best_dist باید دو بعدی باشد")
    best_dist = np.nan_to_num(best_dist, nan=-1).astype(int)

    # آماره‌های توصیفی
    mean_vals = reshape_array(ds[f'{var_name}_mean'].values, n_days, n_points)
    std_vals  = reshape_array(ds[f'{var_name}_std'].values, n_days, n_points)
    skew_vals = reshape_array(ds[f'{var_name}_skewness'].values, n_days, n_points)
    count_vals = reshape_array(ds[f'{var_name}_count'].values, n_days, n_points)

    # ----------------------------------------------------------------
    # ۳-۱. تغییرات روزانه و فصلی
    # ----------------------------------------------------------------
    print("   📊 تحلیل روزانه و فصلی...")
    daily_counts = {d: np.zeros(n_days) for d in dist_codes}
    for day in range(n_days):
        for d in dist_codes:
            daily_counts[d][day] = np.sum(best_dist[day, :] == d)
    daily_percent = {d: daily_counts[d] / n_points * 100 for d in dist_codes}

    # نمودار روزانه (انباشته)
    fig, ax = plt.subplots(figsize=(16, 6))
    days = np.arange(1, n_days + 1)
    bottom = np.zeros(n_days)
    for i, d in enumerate(dist_codes):
        ax.bar(days, daily_percent[d], bottom=bottom,
               label=persian_text(dist_labels[i]), color=dist_colors[i], alpha=0.7, width=1)
        bottom += daily_percent[d]
    ax.set_xlabel(persian_text('روز سال'))
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text(f'تغییرات توزیع غالب در طول سال ({VAR_LABELS.get(var_name, var_name)})'))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_days + 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'daily_distribution_{var_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # تحلیل فصلی
    season_days = {'زمستان': [], 'بهار': [], 'تابستان': [], 'پاییز': []}
    for day in range(n_days):
        season_days[get_season(day)].append(day)

    seasonal_percent = {}
    for season, days_list in season_days.items():
        if days_list:
            seasonal_counts = {d: np.mean([daily_counts[d][day] for day in days_list])
                               for d in dist_codes}
            total = sum(seasonal_counts.values())
            seasonal_percent[season] = {d: (seasonal_counts[d] / total * 100) if total>0 else 0
                                        for d in dist_codes}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(season_days))
    width = 0.2
    for i, d in enumerate(dist_codes):
        values = [seasonal_percent[season][d] if season in seasonal_percent else 0
                  for season in season_days.keys()]
        ax.bar(x + (i - len(dist_codes) / 2 + 0.5) * width, values, width,
               label=persian_text(dist_labels[i]), color=dist_colors[i], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([persian_text(s) for s in season_days.keys()])
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text(f'توزیع تابع‌ها در فصول مختلف ({VAR_LABELS.get(var_name, var_name)})'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'seasonal_distribution_{var_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------
    # ۳-۲. تحلیل ارتفاعی
    # ----------------------------------------------------------------
    print("   📊 تحلیل ارتفاعی...")
    elev_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000]
    elev_labels = ['0-500', '500-1000', '1000-1500', '1500-2000',
                   '2000-2500', '2500-3000', '3000+']
    elev_digitized = np.digitize(elev, elev_bins) - 1
    elev_digitized = np.clip(elev_digitized, 0, len(elev_labels) - 1)

    elev_counts = {}
    for d in dist_codes:
        elev_counts[d] = np.zeros(len(elev_labels))
        for i in range(len(elev_labels)):
            mask = elev_digitized == i
            if np.any(mask):
                elev_counts[d][i] = np.sum(best_dist[:, mask] == d) / (n_days * np.sum(mask)) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(elev_labels))
    width = 0.2
    for i, d in enumerate(dist_codes):
        ax.bar(x + (i - len(dist_codes) / 2 + 0.5) * width, elev_counts[d], width,
               label=persian_text(dist_labels[i]), color=dist_colors[i], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([persian_text(l) for l in elev_labels])
    ax.set_xlabel(persian_text('ارتفاع (متر)'))
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text(f'توزیع تابع‌ها بر اساس ارتفاع ({VAR_LABELS.get(var_name, var_name)})'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'elevation_distribution_{var_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------
    # ۳-۳. نقشه‌های مکانی با درصد میانگین سالانه
    # ----------------------------------------------------------------
    print("   📊 نقشه‌های مکانی...")
    annual_percent = {d: np.mean(daily_percent[d]) for d in dist_codes}
    annual_text_parts = []
    for i, d in enumerate(dist_codes):
        if annual_percent[d] > 0.1:
            label_fa = persian_text(dist_labels[i])
            percent_fa = en_to_fa(annual_percent[d], '%1.1f%%')
            annual_text_parts.append(f"{label_fa} {percent_fa}")
    annual_text = ' | '.join(annual_text_parts)
    title_suffix = persian_text(f"میانگین سالانه: {annual_text}")

    sample_day = n_days // 2
    sample_season = 'تابستان'
    season_days_list = season_days.get(sample_season, [])
    if season_days_list:
        season_best = np.full(n_points, -1, dtype=int)
        for p in range(n_points):
            data = best_dist[season_days_list, p]
            data = data[data >= 0]
            if len(data) > 0:
                counts = np.bincount(data)
                season_best[p] = np.argmax(counts)
    else:
        season_best = None

    day_best = best_dist[sample_day, :]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc1 = axes[0].scatter(lon, lat, c=day_best, cmap='tab10', s=1, alpha=0.6, vmin=-1, vmax=4)
    axes[0].set_title(persian_text(f'روز {sample_day + 1} ({VAR_LABELS.get(var_name, var_name)})') + '\n' + title_suffix)
    axes[0].set_xlabel(persian_text('طول جغرافیایی'))
    axes[0].set_ylabel(persian_text('عرض جغرافیایی'))
    axes[0].grid(True, alpha=0.3)

    if season_best is not None:
        season_counts = {d: 0 for d in dist_codes}
        for day in season_days_list:
            for d in dist_codes:
                season_counts[d] += np.sum(best_dist[day, :] == d)
        total_season = sum(season_counts.values())
        season_percent = {d: (season_counts[d] / total_season * 100) if total_season>0 else 0 for d in dist_codes}
        season_text_parts = []
        for i, d in enumerate(dist_codes):
            if season_percent[d] > 0.1:
                label_fa = persian_text(dist_labels[i])
                percent_fa = en_to_fa(season_percent[d], '%1.1f%%')
                season_text_parts.append(f"{label_fa} {percent_fa}")
        season_text = ' | '.join(season_text_parts)
        season_title = persian_text(f'فصل {sample_season} ({VAR_LABELS.get(var_name, var_name)})') + '\n' + persian_text(f"درصد فصل: {season_text}")

        sc2 = axes[1].scatter(lon, lat, c=season_best, cmap='tab10', s=1, alpha=0.6, vmin=-1, vmax=4)
        axes[1].set_title(season_title)
        axes[1].set_xlabel(persian_text('طول جغرافیایی'))
        axes[1].set_ylabel(persian_text('عرض جغرافیایی'))
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'spatial_distribution_{var_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------
    # ۳-۴. پارامترهای GEV
    # ----------------------------------------------------------------
    print("   📊 تحلیل پارامترهای GEV...")
    gev_vars = [f'{var_name}_gev_p1', f'{var_name}_gev_p2', f'{var_name}_gev_p3']
    if all(v in ds.data_vars for v in gev_vars):
        gev_p1 = reshape_array(ds[gev_vars[0]].values, n_days, n_points)
        gev_p2 = reshape_array(ds[gev_vars[1]].values, n_days, n_points)
        gev_p3 = reshape_array(ds[gev_vars[2]].values, n_days, n_points)

        if gev_p1 is not None and gev_p2 is not None and gev_p3 is not None:
            loc_mean = np.nanmean(gev_p1, axis=1)
            scale_mean = np.nanmean(gev_p2, axis=1)
            shape_mean = np.nanmean(gev_p3, axis=1)

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            axes[0].plot(days, loc_mean, 'b-', linewidth=2)
            axes[0].set_ylabel(persian_text('Location (μ)'))
            axes[0].set_title(persian_text(f'GEV Location - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(days, scale_mean, 'g-', linewidth=2)
            axes[1].set_ylabel(persian_text('Scale (σ)'))
            axes[1].set_title(persian_text(f'GEV Scale - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(days, shape_mean, 'r-', linewidth=2)
            axes[2].set_xlabel(persian_text('روز سال'))
            axes[2].set_ylabel(persian_text('Shape (ξ)'))
            axes[2].set_title(persian_text(f'GEV Shape - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'gev_params_daily_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # تغییرات ارتفاعی
            elev_params = {'location': [], 'scale': [], 'shape': []}
            for i in range(len(elev_labels)):
                mask = elev_digitized == i
                if np.any(mask):
                    elev_params['location'].append(np.nanmean(gev_p1[:, mask]))
                    elev_params['scale'].append(np.nanmean(gev_p2[:, mask]))
                    elev_params['shape'].append(np.nanmean(gev_p3[:, mask]))
                else:
                    elev_params['location'].append(np.nan)
                    elev_params['scale'].append(np.nan)
                    elev_params['shape'].append(np.nan)

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(elev_labels))
            for name, vals in elev_params.items():
                ax.plot(x, vals, marker='o', label=persian_text(name))
            ax.set_xticks(x)
            ax.set_xticklabels([persian_text(l) for l in elev_labels])
            ax.set_xlabel(persian_text('ارتفاع (متر)'))
            ax.set_ylabel(persian_text('مقدار پارامتر'))
            ax.set_title(persian_text(f'تغییرات پارامترهای GEV با ارتفاع ({VAR_LABELS.get(var_name, var_name)})'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'gev_params_elevation_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("   ⚠️ متغیرهای GEV یافت نشد.")

    # ----------------------------------------------------------------
    # ۳-۵. پارامترهای Skew‑Normal
    # ----------------------------------------------------------------
    print("   📊 تحلیل پارامترهای Skew‑Normal...")
    skew_vars = [f'{var_name}_skew_p1', f'{var_name}_skew_p2', f'{var_name}_skew_p3']
    if all(v in ds.data_vars for v in skew_vars):
        skew_p1 = reshape_array(ds[skew_vars[0]].values, n_days, n_points)
        skew_p2 = reshape_array(ds[skew_vars[1]].values, n_days, n_points)
        skew_p3 = reshape_array(ds[skew_vars[2]].values, n_days, n_points)
        if skew_p1 is not None and skew_p2 is not None and skew_p3 is not None:
            loc_mean = np.nanmean(skew_p1, axis=1)
            scale_mean = np.nanmean(skew_p2, axis=1)
            shape_mean = np.nanmean(skew_p3, axis=1)

            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            axes[0].plot(days, loc_mean, 'c-', linewidth=2)
            axes[0].set_ylabel(persian_text('Location'))
            axes[0].set_title(persian_text(f'Skew‑Normal Location - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(days, scale_mean, 'm-', linewidth=2)
            axes[1].set_ylabel(persian_text('Scale'))
            axes[1].set_title(persian_text(f'Skew‑Normal Scale - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(days, shape_mean, 'y-', linewidth=2)
            axes[2].set_xlabel(persian_text('روز سال'))
            axes[2].set_ylabel(persian_text('Shape'))
            axes[2].set_title(persian_text(f'Skew‑Normal Shape - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'skew_normal_params_daily_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("   ⚠️ متغیرهای Skew‑Normal یافت نشد.")

    # ----------------------------------------------------------------
    # ۳-۶. پارامترهای Bimodal
    # ----------------------------------------------------------------
    print("   📊 تحلیل پارامترهای Bimodal...")
    bimodal_vars = [f'{var_name}_bimodal_p1', f'{var_name}_bimodal_p2', f'{var_name}_bimodal_p3',
                    f'{var_name}_bimodal_p4', f'{var_name}_bimodal_p5']
    if all(v in ds.data_vars for v in bimodal_vars):
        bimodal_p1 = reshape_array(ds[bimodal_vars[0]].values, n_days, n_points)
        bimodal_p2 = reshape_array(ds[bimodal_vars[1]].values, n_days, n_points)
        bimodal_p3 = reshape_array(ds[bimodal_vars[2]].values, n_days, n_points)
        bimodal_p4 = reshape_array(ds[bimodal_vars[3]].values, n_days, n_points)
        bimodal_p5 = reshape_array(ds[bimodal_vars[4]].values, n_days, n_points)
        if all(x is not None for x in [bimodal_p1, bimodal_p2, bimodal_p3, bimodal_p4, bimodal_p5]):
            w1_mean = np.nanmean(bimodal_p1, axis=1)
            mu1_mean = np.nanmean(bimodal_p2, axis=1)
            sigma1_mean = np.nanmean(bimodal_p3, axis=1)
            mu2_mean = np.nanmean(bimodal_p4, axis=1)
            sigma2_mean = np.nanmean(bimodal_p5, axis=1)

            fig, axes = plt.subplots(5, 1, figsize=(14, 14))
            axes[0].plot(days, w1_mean, 'k-', linewidth=2)
            axes[0].set_ylabel(persian_text('w1'))
            axes[0].set_title(persian_text(f'Bimodal w1 - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(days, mu1_mean, 'b-', linewidth=2)
            axes[1].set_ylabel(persian_text('μ1'))
            axes[1].set_title(persian_text(f'Bimodal μ1 - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[1].grid(True, alpha=0.3)
            axes[2].plot(days, sigma1_mean, 'g-', linewidth=2)
            axes[2].set_ylabel(persian_text('σ1'))
            axes[2].set_title(persian_text(f'Bimodal σ1 - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[2].grid(True, alpha=0.3)
            axes[3].plot(days, mu2_mean, 'r-', linewidth=2)
            axes[3].set_ylabel(persian_text('μ2'))
            axes[3].set_title(persian_text(f'Bimodal μ2 - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[3].grid(True, alpha=0.3)
            axes[4].plot(days, sigma2_mean, 'm-', linewidth=2)
            axes[4].set_xlabel(persian_text('روز سال'))
            axes[4].set_ylabel(persian_text('σ2'))
            axes[4].set_title(persian_text(f'Bimodal σ2 - روزانه ({VAR_LABELS.get(var_name, var_name)})'))
            axes[4].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'bimodal_params_daily_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print("   ⚠️ متغیرهای Bimodal یافت نشد.")

    # ----------------------------------------------------------------
    # ۳-۷. همبستگی نوع توزیع با آماره‌های توصیفی
    # ----------------------------------------------------------------
    print("   📊 همبستگی نوع توزیع با آماره‌ها...")
    mode_dist = np.full(n_points, -1, dtype=int)
    for p in range(n_points):
        data = best_dist[:, p]
        data = data[data >= 0]
        if len(data) > 0:
            counts = np.bincount(data)
            mode_dist[p] = np.argmax(counts)

    mean_flat = to_1d(mean_vals, n_points) if mean_vals is not None else np.full(n_points, np.nan)
    std_flat  = to_1d(std_vals, n_points) if std_vals is not None else np.full(n_points, np.nan)
    skew_flat = to_1d(skew_vals, n_points) if skew_vals is not None else np.full(n_points, np.nan)
    elev_flat = to_1d(elev, n_points)

    df_points = pd.DataFrame({
        'dist': mode_dist,
        'mean': mean_flat,
        'std': std_flat,
        'skew': skew_flat,
        'elev': elev_flat
    })
    df_points = df_points[(df_points['dist'] >= 0) & (~np.isnan(df_points['mean']))]

    if len(df_points) > 0:
        stats_by_dist = df_points.groupby('dist').agg({
            'mean': ['mean', 'std'],
            'std': ['mean', 'std'],
            'skew': ['mean', 'std'],
            'elev': ['mean', 'std']
        }).round(2)
        stats_by_dist.index = [dist_labels[i] if i < len(dist_labels) else f'Dist{i}'
                               for i in stats_by_dist.index]
        stats_by_dist.to_csv(os.path.join(output_dir, f'stats_by_distribution_{var_name}.csv'))
        print(f"   ✅ stats_by_distribution_{var_name}.csv")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col in zip(axes, ['mean', 'std', 'skew']):
            df_points.boxplot(column=col, by='dist', ax=ax)
            ax.set_title(persian_text(f'{col} بر حسب توزیع ({VAR_LABELS.get(var_name, var_name)})'))
            ax.set_xlabel(persian_text('نوع توزیع'))
            ax.grid(True, alpha=0.3)
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplots_by_distribution_{var_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("   ⚠️ داده‌ای برای همبستگی وجود ندارد.")

    # ----------------------------------------------------------------
    # ۳-۸. خوشه‌بندی
    # ----------------------------------------------------------------
    print("   📊 خوشه‌بندی ایستگاه‌ها...")
    features = []
    if mean_vals is not None:
        features.append(to_1d(mean_vals, n_points))
    if std_vals is not None:
        features.append(to_1d(std_vals, n_points))
    if skew_vals is not None:
        features.append(to_1d(skew_vals, n_points))
    if elev is not None:
        features.append(to_1d(elev, n_points))

    if len(features) >= 2:
        X = np.column_stack(features)
        nan_mask = np.isnan(X).any(axis=1)
        X_clean = X[~nan_mask]
        if len(X_clean) > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            inertias = []
            K_range = range(2, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(K_range, inertias, 'bo-')
            ax.set_xlabel(persian_text('تعداد خوشه‌ها'))
            ax.set_ylabel(persian_text('اینرسی'))
            ax.set_title(persian_text(f'نمودار Elbow ({VAR_LABELS.get(var_name, var_name)})'))
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'elbow_plot_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            valid_indices = np.where(~nan_mask)[0]
            cluster_dist = np.zeros((n_clusters, len(dist_codes)))
            for idx, orig_idx in enumerate(valid_indices):
                c = clusters[idx]
                d = mode_dist[orig_idx]
                if d >= 0:
                    cluster_dist[c, d] += 1
            cluster_percent = cluster_dist / cluster_dist.sum(axis=1, keepdims=True) * 100
            df_cluster = pd.DataFrame(cluster_percent, columns=dist_labels)
            df_cluster['خوشه'] = range(1, n_clusters + 1)
            df_cluster = df_cluster.set_index('خوشه')
            df_cluster.to_csv(os.path.join(output_dir, f'cluster_distribution_{var_name}.csv'))
            print(f"   ✅ cluster_distribution_{var_name}.csv")

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(n_clusters)
            bottom = np.zeros(n_clusters)
            for i, d in enumerate(dist_codes):
                ax.bar(x, cluster_percent[:, i], bottom=bottom,
                       label=persian_text(dist_labels[i]), color=dist_colors[i], alpha=0.7)
                bottom += cluster_percent[:, i]
            ax.set_xticks(x)
            ax.set_xticklabels([persian_text(f'خوشه {i + 1}') for i in range(n_clusters)])
            ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
            ax.set_title(persian_text(f'ترکیب خوشه‌ها ({VAR_LABELS.get(var_name, var_name)})'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cluster_composition_{var_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("   ⚠️ داده‌ای برای خوشه‌بندی وجود ندارد.")
    else:
        print("   ⚠️ ویژگی‌های کافی برای خوشه‌بندی موجود نیست.")

    # ----------------------------------------------------------------
    # ۳-۹. مقایسه معیارهای اطلاعاتی
    # ----------------------------------------------------------------
    print("   📊 مقایسه معیارهای اطلاعاتی (AIC و BIC)...")
    dist_list = ['normal', 'skew', 'gev', 'bimodal', 'pearson']
    aic_dict = {}
    bic_dict = {}
    for dist in dist_list:
        aic_var = f'{var_name}_{dist}_aicc'
        bic_var = f'{var_name}_{dist}_bic'
        if aic_var in ds.data_vars:
            arr = ds[aic_var].values
            if arr.ndim > 1:
                arr = arr.flatten()
            aic_dict[dist_labels[dist_codes[dist_list.index(dist)]]] = np.nanmean(arr)
        if bic_var in ds.data_vars:
            arr = ds[bic_var].values
            if arr.ndim > 1:
                arr = arr.flatten()
            bic_dict[dist_labels[dist_codes[dist_list.index(dist)]]] = np.nanmean(arr)

    if aic_dict:
        df_aic = pd.DataFrame(list(aic_dict.items()), columns=['توزیع', 'میانگین AICc'])
        df_aic = df_aic.sort_values('میانگین AICc')
        df_aic.to_csv(os.path.join(output_dir, f'aic_comparison_{var_name}.csv'), index=False)
        print(f"   ✅ aic_comparison_{var_name}.csv")
    if bic_dict:
        df_bic = pd.DataFrame(list(bic_dict.items()), columns=['توزیع', 'میانگین BIC'])
        df_bic = df_bic.sort_values('میانگین BIC')
        df_bic.to_csv(os.path.join(output_dir, f'bic_comparison_{var_name}.csv'), index=False)
        print(f"   ✅ bic_comparison_{var_name}.csv")

    # بازگرداندن نتایج روزانه و فصلی برای استفاده در مقایسه‌ها
    return {
        'daily_percent': daily_percent,
        'seasonal_percent': seasonal_percent,
        'elev_counts': elev_counts,
        'annual_percent': annual_percent,
        'best_dist': best_dist,
        'mode_dist': mode_dist,
        'aic_dict': aic_dict,
        'bic_dict': bic_dict
    }

# ============================================================================
# ۴. اجرای تحلیل برای هر سه متغیر
# ============================================================================
results = {}
for var in VARS:
    try:
        res = analyze_single_variable(var, ds, n_days, n_points, day_dim, point_dim,
                                      elev, lat, lon, OUTPUT_DIR)
        results[var] = res
    except Exception as e:
        print(f"❌ خطا در تحلیل {var}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ۵. بخش مقایسه‌ای بین سه متغیر
# ============================================================================
if len(results) >= 2:
    print("\n" + "="*80)
    print("🔹 مقایسه‌ی مستقیم بین متغیرهای دمایی")
    print("="*80)

    # ۵-۱. مقایسه‌ی درصد توزیع‌ها در کل سال (میانگین سالانه)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dist_labels))
    width = 0.25
    for i, var in enumerate(VARS):
        if var in results:
            annual = results[var]['annual_percent']
            values = [annual.get(d, 0) for d in dist_codes]
            ax.bar(x + (i - (len(VARS)-1)/2) * width, values, width,
                   label=VAR_LABELS.get(var, var), color=VAR_COLORS.get(var, 'gray'), alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([persian_text(l) for l in dist_labels])
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text('مقایسه‌ی میانگین سالانه‌ی توزیع‌های غالب در سه متغیر دمایی'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_annual_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ comparison_annual_distributions.png")

    # ۵-۲. مقایسه‌ی فصلی (برای یک فصل خاص مثلاً تابستان)
    season_name = 'تابستان'
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dist_labels))
    width = 0.25
    for i, var in enumerate(VARS):
        if var in results and season_name in results[var]['seasonal_percent']:
            seasonal = results[var]['seasonal_percent'][season_name]
            values = [seasonal.get(d, 0) for d in dist_codes]
            ax.bar(x + (i - (len(VARS)-1)/2) * width, values, width,
                   label=VAR_LABELS.get(var, var), color=VAR_COLORS.get(var, 'gray'), alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([persian_text(l) for l in dist_labels])
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text(f'مقایسه‌ی توزیع‌ها در فصل {season_name}'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'comparison_seasonal_{season_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ comparison_seasonal_{season_name}.png")

    # ۵-۳. مقایسه‌ی ارتفاعی (برای یک بازه‌ی ارتفاعی مثلاً ۱۰۰۰-۱۵۰۰ متر)
    elev_index = 2  # 1000-1500 متر
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dist_labels))
    width = 0.25
    for i, var in enumerate(VARS):
        if var in results:
            elev_counts = results[var]['elev_counts']
            values = [elev_counts[d][elev_index] if elev_index < len(elev_counts[d]) else 0 for d in dist_codes]
            ax.bar(x + (i - (len(VARS)-1)/2) * width, values, width,
                   label=VAR_LABELS.get(var, var), color=VAR_COLORS.get(var, 'gray'), alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([persian_text(l) for l in dist_labels])
    ax.set_ylabel(persian_text('درصد ایستگاه‌ها (%)'))
    ax.set_title(persian_text('مقایسه‌ی توزیع‌ها در ارتفاع ۱۰۰۰-۱۵۰۰ متر'))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_elevation_1000-1500.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ comparison_elevation_1000-1500.png")

    # ۵-۴. مقایسه‌ی AICc بین سه متغیر (میانگین کلی)
    aic_df_list = []
    for var in VARS:
        if var in results and results[var]['aic_dict']:
            df = pd.DataFrame({
                'متغیر': VAR_LABELS.get(var, var),
                'توزیع': list(results[var]['aic_dict'].keys()),
                'AICc': list(results[var]['aic_dict'].values())
            })
            aic_df_list.append(df)
    if aic_df_list:
        aic_all = pd.concat(aic_df_list, ignore_index=True)
        # برای هر متغیر، توزیع برتر را پیدا می‌کنیم
        best_aic = aic_all.loc[aic_all.groupby('متغیر')['AICc'].idxmin()]
        print("\nبهترین توزیع از نظر AICc برای هر متغیر:")
        print(best_aic[['متغیر', 'توزیع', 'AICc']].to_string(index=False))
        best_aic.to_csv(os.path.join(OUTPUT_DIR, 'comparison_best_aicc.csv'), index=False)
        print("   ✅ comparison_best_aicc.csv")

        # نمودار مقایسه‌ای AICc برای همه‌ی توزیع‌ها و متغیرها
        fig, ax = plt.subplots(figsize=(12, 6))
        for var in VARS:
            if var in results and results[var]['aic_dict']:
                aic_vals = list(results[var]['aic_dict'].values())
                ax.plot(dist_labels, aic_vals, marker='o', label=VAR_LABELS.get(var, var),
                        color=VAR_COLORS.get(var, 'gray'), linewidth=2)
        ax.set_xlabel(persian_text('نوع توزیع'))
        ax.set_ylabel('میانگین AICc')
        ax.set_title(persian_text('مقایسه‌ی AICc بین سه متغیر دمایی'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_aicc.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ comparison_aicc.png")

    # ۵-۵. جدول جامع مقایسه‌ای (میانگین درصد هر توزیع در کل سال)
    comp_data = []
    for var in VARS:
        if var in results:
            annual = results[var]['annual_percent']
            row = {'متغیر': VAR_LABELS.get(var, var)}
            for i, label in enumerate(dist_labels):
                row[label] = annual.get(i, 0)
            comp_data.append(row)
    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(os.path.join(OUTPUT_DIR, 'comparison_annual_percent_table.csv'), index=False)
    print("   ✅ comparison_annual_percent_table.csv")
    print("\nجدول مقایسه‌ی درصد سالانه توزیع‌ها:")
    print(df_comp.to_string(index=False))

# ============================================================================
# ۶. جمع‌بندی نهایی
# ============================================================================
print("\n" + "="*80)
print("✅ تحلیل کامل شد. تمام خروجی‌ها در پوشهٔ زیر ذخیره شدند:")
print(f"   {OUTPUT_DIR}")
print("="*80)

ds.close()
