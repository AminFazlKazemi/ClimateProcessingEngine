#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_distributions_final.py
================================================================================
تحلیل جامع توزیع‌های برازش‌شده (خروجی ۲۹ متغیری) – نسخه نهایی
================================================================================
- خواندن فایل Zarr اقلیم‌شناسی (خروجی نهایی پروژه)
- استفاده از ۵ توزیع: Normal, Skew-Normal, GEV, Bimodal, Pearson
- تحلیل روزانه، فصلی، ارتفاعی و مکانی
- بررسی پارامترهای هر توزیع (GEV, Skew‑Normal, Bimodal)
- خوشه‌بندی ایستگاه‌ها بر اساس ویژگی‌ها
- خروجی نمودارها، نقشه‌ها و جداول جامع
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
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# ============================================================================
# ۰. انتخاب متغیر مورد نظر (tmax / tmean / tmin)
# ============================================================================
VAR = 'tmax'   # ← این را به 'tmean' یا 'tmin' تغییر دهید

# ============================================================================
# ۱. تنظیمات و بارگذاری داده‌ها
# ============================================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"

print("=" * 80)
print(f"📊 تحلیل جامع توزیع‌های برازش‌شده (متغیر: {VAR})")
print("=" * 80)

if not os.path.exists(ZARR_PATH):
    print(f"❌ فایل Zarr یافت نشد: {ZARR_PATH}")
    sys.exit()

# ============================================================
# نکته مهم: consolidated=False چون فایل با متادیتای یکپارچه ذخیره نشده
# ============================================================
ds = xr.open_zarr(ZARR_PATH, consolidated=False)
print(f"✅ فایل بارگذاری شد. ابعاد: {list(ds.dims)}")
print(f"   متغیرها: {list(ds.data_vars)}")

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

# ============================================================================
# ۲. استخراج داده‌های اصلی با پیشوند VAR
# ============================================================================

best_dist = ds[f'{VAR}_best_dist'].values
if best_dist.ndim == 2:
    if best_dist.shape[0] != n_days or best_dist.shape[1] != n_points:
        if best_dist.shape[0] == n_points and best_dist.shape[1] == n_days:
            best_dist = best_dist.T
        else:
            best_dist = best_dist.reshape(n_days, n_points)

# کدهای توزیع (۵ توزیع)
dist_codes = [0, 1, 2, 3, 4]
dist_labels = ['Normal', 'Skew-Normal', 'GEV', 'Bimodal', 'Pearson']
dist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# آماره‌های توصیفی
mean_vals = ds[f'{VAR}_mean'].values
std_vals = ds[f'{VAR}_std'].values
skew_vals = ds[f'{VAR}_skewness'].values
count_vals = ds[f'{VAR}_count'].values

# تبدیل به شکل (day, point) در صورت لزوم
for arr in [mean_vals, std_vals, skew_vals, count_vals]:
    if arr.ndim == 1:
        if arr.size == n_points:
            arr = np.tile(arr, (n_days, 1))
        elif arr.size == n_days:
            arr = np.tile(arr, (n_points, 1)).T
    elif arr.ndim == 2:
        if arr.shape[0] != n_days or arr.shape[1] != n_points:
            if arr.shape[0] == n_points and arr.shape[1] == n_days:
                arr = arr.T

# مختصات
elev = ds['elev'].values
if elev.ndim > 1:
    elev = elev.flatten()
lat = ds['lat'].values
if lat.ndim > 1:
    lat = lat.flatten()
lon = ds['lon'].values
if lon.ndim > 1:
    lon = lon.flatten()

# ============================================================================
# ۳. توابع کمکی
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
    """تبدیل آرایه به یک‌بعدی با طول مشخص"""
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

# ============================================================================
# ۴. تحلیل اصلی
# ============================================================================

print("\n" + "=" * 60)
print("🔹 تحلیل توزیع‌های برازش‌شده")
print("=" * 60)

# ----------------------------------------------------------------
# ۴-۱. تغییرات روزانه و فصلی
# ----------------------------------------------------------------
print("\n📊 تحلیل تغییرات روزانه و فصلی...")

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
           label=dist_labels[i], color=dist_colors[i], alpha=0.7, width=1)
    bottom += daily_percent[d]
ax.set_xlabel('روز سال')
ax.set_ylabel('درصد ایستگاه‌ها (%)')
ax.set_title(f'تغییرات توزیع غالب در طول سال ({VAR})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, n_days + 1)
plt.tight_layout()
plt.savefig(f'daily_distribution_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ daily_distribution_{VAR}.png")

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
        seasonal_percent[season] = {d: seasonal_counts[d] / total * 100
                                    for d in dist_codes}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(season_days))
width = 0.2
for i, d in enumerate(dist_codes):
    values = [seasonal_percent[season][d] if season in seasonal_percent else 0
              for season in season_days.keys()]
    ax.bar(x + (i - len(dist_codes) / 2 + 0.5) * width, values, width,
           label=dist_labels[i], color=dist_colors[i], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(season_days.keys())
ax.set_ylabel('درصد ایستگاه‌ها (%)')
ax.set_title(f'توزیع تابع‌ها در فصول مختلف ({VAR})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'seasonal_distribution_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ seasonal_distribution_{VAR}.png")

# ----------------------------------------------------------------
# ۴-۲. تحلیل ارتفاعی
# ----------------------------------------------------------------
print("\n📊 تحلیل ارتفاعی...")

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
           label=dist_labels[i], color=dist_colors[i], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(elev_labels)
ax.set_xlabel('ارتفاع (متر)')
ax.set_ylabel('درصد ایستگاه‌ها (%)')
ax.set_title(f'توزیع تابع‌ها بر اساس ارتفاع ({VAR})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'elevation_distribution_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ elevation_distribution_{VAR}.png")

# ----------------------------------------------------------------
# ۴-۳. نقشه‌های مکانی
# ----------------------------------------------------------------
print("\n📊 نقشه‌های مکانی...")

sample_day = n_days // 2  # روز میانی سال
sample_season = 'تابستان'
season_days_list = season_days.get(sample_season, [])
if season_days_list:
    season_best = np.zeros(n_points, dtype=int)
    for p in range(n_points):
        counts = np.bincount(best_dist[season_days_list, p].astype(int))
        if len(counts) > 0:
            season_best[p] = np.argmax(counts)
        else:
            season_best[p] = -1
else:
    season_best = None

day_best = best_dist[sample_day, :]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sc1 = axes[0].scatter(lon, lat, c=day_best, cmap='tab10', s=1, alpha=0.6, vmin=-1, vmax=4)
axes[0].set_title(f'روز {sample_day + 1} ({VAR})')
axes[0].set_xlabel('طول جغرافیایی')
axes[0].set_ylabel('عرض جغرافیایی')
axes[0].grid(True, alpha=0.3)

if season_best is not None:
    sc2 = axes[1].scatter(lon, lat, c=season_best, cmap='tab10', s=1, alpha=0.6, vmin=-1, vmax=4)
    axes[1].set_title(f'فصل {sample_season} ({VAR})')
    axes[1].set_xlabel('طول جغرافیایی')
    axes[1].set_ylabel('عرض جغرافیایی')
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'spatial_distribution_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ spatial_distribution_{VAR}.png")

# ----------------------------------------------------------------
# ۴-۴. پارامترهای GEV
# ----------------------------------------------------------------
print("\n📊 تحلیل پارامترهای GEV...")

gev_vars = [f'{VAR}_gev_p1', f'{VAR}_gev_p2', f'{VAR}_gev_p3']
if all(v in ds.data_vars for v in gev_vars):
    gev_p1 = ds[f'{VAR}_gev_p1'].values
    gev_p2 = ds[f'{VAR}_gev_p2'].values
    gev_p3 = ds[f'{VAR}_gev_p3'].values

    for arr in [gev_p1, gev_p2, gev_p3]:
        if arr.ndim == 1:
            if arr.size == n_points:
                arr = np.tile(arr, (n_days, 1))
            elif arr.size == n_days:
                arr = np.tile(arr, (n_points, 1)).T
        elif arr.ndim == 2:
            if arr.shape[0] != n_days or arr.shape[1] != n_points:
                if arr.shape[0] == n_points and arr.shape[1] == n_days:
                    arr = arr.T

    loc_mean = np.nanmean(gev_p1, axis=1)
    scale_mean = np.nanmean(gev_p2, axis=1)
    shape_mean = np.nanmean(gev_p3, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(days, loc_mean, 'b-', linewidth=2)
    axes[0].set_ylabel('Location (μ)')
    axes[0].set_title(f'GEV Location - روزانه ({VAR})')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(days, scale_mean, 'g-', linewidth=2)
    axes[1].set_ylabel('Scale (σ)')
    axes[1].set_title(f'GEV Scale - روزانه ({VAR})')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(days, shape_mean, 'r-', linewidth=2)
    axes[2].set_xlabel('روز سال')
    axes[2].set_ylabel('Shape (ξ)')
    axes[2].set_title(f'GEV Shape - روزانه ({VAR})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gev_params_daily_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ gev_params_daily_{VAR}.png")

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
        ax.plot(x, vals, marker='o', label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(elev_labels)
    ax.set_xlabel('ارتفاع (متر)')
    ax.set_ylabel('مقدار پارامتر')
    ax.set_title(f'تغییرات پارامترهای GEV با ارتفاع ({VAR})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'gev_params_elevation_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ gev_params_elevation_{VAR}.png")
else:
    print("   ⚠️ متغیرهای GEV یافت نشد.")

# ----------------------------------------------------------------
# ۴-۵. پارامترهای Skew-Normal
# ----------------------------------------------------------------
print("\n📊 تحلیل پارامترهای Skew-Normal...")

skew_vars = [f'{VAR}_skew_p1', f'{VAR}_skew_p2', f'{VAR}_skew_p3']
if all(v in ds.data_vars for v in skew_vars):
    skew_p1 = ds[f'{VAR}_skew_p1'].values
    skew_p2 = ds[f'{VAR}_skew_p2'].values
    skew_p3 = ds[f'{VAR}_skew_p3'].values

    for arr in [skew_p1, skew_p2, skew_p3]:
        if arr.ndim == 1:
            if arr.size == n_points:
                arr = np.tile(arr, (n_days, 1))
            elif arr.size == n_days:
                arr = np.tile(arr, (n_points, 1)).T
        elif arr.ndim == 2:
            if arr.shape[0] != n_days or arr.shape[1] != n_points:
                if arr.shape[0] == n_points and arr.shape[1] == n_days:
                    arr = arr.T

    loc_mean = np.nanmean(skew_p1, axis=1)
    scale_mean = np.nanmean(skew_p2, axis=1)
    shape_mean = np.nanmean(skew_p3, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(days, loc_mean, 'c-', linewidth=2)
    axes[0].set_ylabel('Location')
    axes[0].set_title(f'Skew-Normal Location - روزانه ({VAR})')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(days, scale_mean, 'm-', linewidth=2)
    axes[1].set_ylabel('Scale')
    axes[1].set_title(f'Skew-Normal Scale - روزانه ({VAR})')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(days, shape_mean, 'y-', linewidth=2)
    axes[2].set_xlabel('روز سال')
    axes[2].set_ylabel('Shape')
    axes[2].set_title(f'Skew-Normal Shape - روزانه ({VAR})')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'skew_normal_params_daily_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ skew_normal_params_daily_{VAR}.png")
else:
    print("   ⚠️ متغیرهای Skew-Normal یافت نشد.")

# ----------------------------------------------------------------
# ۴-۶. پارامترهای Bimodal (اختیاری)
# ----------------------------------------------------------------
print("\n📊 تحلیل پارامترهای Bimodal...")

bimodal_vars = [f'{VAR}_bimodal_p1', f'{VAR}_bimodal_p2', f'{VAR}_bimodal_p3',
                f'{VAR}_bimodal_p4', f'{VAR}_bimodal_p5']
if all(v in ds.data_vars for v in bimodal_vars):
    bimodal_p1 = ds[f'{VAR}_bimodal_p1'].values
    bimodal_p2 = ds[f'{VAR}_bimodal_p2'].values
    bimodal_p3 = ds[f'{VAR}_bimodal_p3'].values
    bimodal_p4 = ds[f'{VAR}_bimodal_p4'].values
    bimodal_p5 = ds[f'{VAR}_bimodal_p5'].values

    for arr in [bimodal_p1, bimodal_p2, bimodal_p3, bimodal_p4, bimodal_p5]:
        if arr.ndim == 1:
            if arr.size == n_points:
                arr = np.tile(arr, (n_days, 1))
            elif arr.size == n_days:
                arr = np.tile(arr, (n_points, 1)).T
        elif arr.ndim == 2:
            if arr.shape[0] != n_days or arr.shape[1] != n_points:
                if arr.shape[0] == n_points and arr.shape[1] == n_days:
                    arr = arr.T

    # میانگین روزانه پارامترها
    w1_mean = np.nanmean(bimodal_p1, axis=1)
    mu1_mean = np.nanmean(bimodal_p2, axis=1)
    sigma1_mean = np.nanmean(bimodal_p3, axis=1)
    mu2_mean = np.nanmean(bimodal_p4, axis=1)
    sigma2_mean = np.nanmean(bimodal_p5, axis=1)

    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    axes[0].plot(days, w1_mean, 'k-', linewidth=2)
    axes[0].set_ylabel('w1')
    axes[0].set_title(f'Bimodal w1 - روزانه ({VAR})')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(days, mu1_mean, 'b-', linewidth=2)
    axes[1].set_ylabel('μ1')
    axes[1].set_title(f'Bimodal μ1 - روزانه ({VAR})')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(days, sigma1_mean, 'g-', linewidth=2)
    axes[2].set_ylabel('σ1')
    axes[2].set_title(f'Bimodal σ1 - روزانه ({VAR})')
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(days, mu2_mean, 'r-', linewidth=2)
    axes[3].set_ylabel('μ2')
    axes[3].set_title(f'Bimodal μ2 - روزانه ({VAR})')
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(days, sigma2_mean, 'm-', linewidth=2)
    axes[4].set_xlabel('روز سال')
    axes[4].set_ylabel('σ2')
    axes[4].set_title(f'Bimodal σ2 - روزانه ({VAR})')
    axes[4].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'bimodal_params_daily_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ bimodal_params_daily_{VAR}.png")
else:
    print("   ⚠️ متغیرهای Bimodal یافت نشد.")

# ----------------------------------------------------------------
# ۴-۷. همبستگی نوع توزیع با آماره‌های توصیفی
# ----------------------------------------------------------------
print("\n📊 همبستگی نوع توزیع با آماره‌های توصیفی...")

mode_dist = np.zeros(n_points, dtype=int)
for p in range(n_points):
    counts = np.bincount(best_dist[:, p].astype(int))
    if len(counts) > 0:
        mode_dist[p] = np.argmax(counts)
    else:
        mode_dist[p] = -1

mean_flat = to_1d(mean_vals, n_points)
std_flat = to_1d(std_vals, n_points)
skew_flat = to_1d(skew_vals, n_points)
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
    print("\nجدول آماره‌ها بر اساس نوع توزیع غالب:")
    print(stats_by_dist)
    stats_by_dist.to_csv(f'stats_by_distribution_{VAR}.csv')
    print(f"   ✅ stats_by_distribution_{VAR}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ['mean', 'std', 'skew']):
        df_points.boxplot(column=col, by='dist', ax=ax)
        ax.set_title(f'{col} بر حسب توزیع ({VAR})')
        ax.set_xlabel('نوع توزیع')
        ax.grid(True, alpha=0.3)
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(f'boxplots_by_distribution_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ boxplots_by_distribution_{VAR}.png")
else:
    print("   ⚠️ داده‌ای برای همبستگی وجود ندارد.")

# ----------------------------------------------------------------
# ۴-۸. خوشه‌بندی
# ----------------------------------------------------------------
print("\n📊 خوشه‌بندی ایستگاه‌ها بر اساس ویژگی‌ها...")

features = []
if mean_vals is not None:
    features.append(to_1d(mean_vals, n_points))
if std_vals is not None:
    features.append(to_1d(std_vals, n_points))
if skew_vals is not None:
    features.append(to_1d(skew_vals, n_points))
if elev is not None:
    features.append(to_1d(elev, n_points))

if features:
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
        ax.set_xlabel('تعداد خوشه‌ها')
        ax.set_ylabel('اینرسی')
        ax.set_title(f'نمودار Elbow ({VAR})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'elbow_plot_{VAR}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ elbow_plot_{VAR}.png")

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
        print("\nترکیب توزیع‌ها در خوشه‌ها:")
        print(df_cluster)
        df_cluster.to_csv(f'cluster_distribution_{VAR}.csv')
        print(f"   ✅ cluster_distribution_{VAR}.csv")

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(n_clusters)
        bottom = np.zeros(n_clusters)
        for i, d in enumerate(dist_codes):
            ax.bar(x, cluster_percent[:, i], bottom=bottom,
                   label=dist_labels[i], color=dist_colors[i], alpha=0.7)
            bottom += cluster_percent[:, i]
        ax.set_xticks(x)
        ax.set_xticklabels([f'خوشه {i + 1}' for i in range(n_clusters)])
        ax.set_ylabel('درصد ایستگاه‌ها (%)')
        ax.set_title(f'ترکیب خوشه‌ها ({VAR})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'cluster_composition_{VAR}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ cluster_composition_{VAR}.png")
    else:
        print("   ⚠️ داده‌ای برای خوشه‌بندی وجود ندارد.")
else:
    print("   ⚠️ ویژگی‌های کافی برای خوشه‌بندی موجود نیست.")

# ----------------------------------------------------------------
# ۴-۹. مقایسه معیارهای اطلاعاتی (AIC و BIC)
# ----------------------------------------------------------------
print("\n📊 مقایسه معیارهای اطلاعاتی (AIC و BIC)...")

dist_list = ['normal', 'skew', 'gev', 'bimodal', 'pearson']
aic_dict = {}
bic_dict = {}
for dist in dist_list:
    aic_var = f'{VAR}_{dist}_aicc'
    bic_var = f'{VAR}_{dist}_bic'
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
    print("\nمیانگین AICc:")
    print(df_aic.to_string(index=False))
    df_aic.to_csv(f'aic_comparison_{VAR}.csv', index=False)
    print(f"   ✅ aic_comparison_{VAR}.csv")

if bic_dict:
    df_bic = pd.DataFrame(list(bic_dict.items()), columns=['توزیع', 'میانگین BIC'])
    df_bic = df_bic.sort_values('میانگین BIC')
    print("\nمیانگین BIC:")
    print(df_bic.to_string(index=False))
    df_bic.to_csv(f'bic_comparison_{VAR}.csv', index=False)
    print(f"   ✅ bic_comparison_{VAR}.csv")

# ============================================================================
# ۵. جمع‌بندی نهایی
# ============================================================================

print("\n" + "=" * 80)
print(f"✅ تحلیل کامل شد (متغیر: {VAR})")
print("📁 فایل‌های خروجی:")
files = [
    f'daily_distribution_{VAR}.png',
    f'seasonal_distribution_{VAR}.png',
    f'elevation_distribution_{VAR}.png',
    f'spatial_distribution_{VAR}.png',
    f'gev_params_daily_{VAR}.png',
    f'gev_params_elevation_{VAR}.png',
    f'skew_normal_params_daily_{VAR}.png',
    f'bimodal_params_daily_{VAR}.png',
    f'stats_by_distribution_{VAR}.csv',
    f'cluster_distribution_{VAR}.csv',
    f'boxplots_by_distribution_{VAR}.png',
    f'elbow_plot_{VAR}.png',
    f'cluster_composition_{VAR}.png',
    f'aic_comparison_{VAR}.csv',
    f'bic_comparison_{VAR}.csv'
]
for f in files:
    print(f"   - {f}")
print("=" * 80)

ds.close()