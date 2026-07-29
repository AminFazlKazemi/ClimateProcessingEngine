#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_gof.py
================================================================================
مقایسه آماره‌های نیکویی برازش برای ۵ توزیع (Normal, Skew‑Normal, GEV, Bimodal, Pearson)
با استفاده از داده‌های موجود در فایل Zarr خروجی.
================================================================================
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# تنظیمات
# ============================================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
VAR = 'tmax'   # یا 'tmean' یا 'tmin'

print("=" * 80)
print(f"📊 مقایسه آماره‌های نیکویی برازش (متغیر: {VAR})")
print("=" * 80)

# بارگذاری داده
ds = xr.open_zarr(ZARR_PATH, consolidated=False)

# نام توزیع‌ها و کدها
dist_names = ['normal', 'skew', 'gev', 'bimodal', 'pearson']
dist_labels = ['Normal', 'Skew‑Normal', 'GEV', 'Bimodal', 'Pearson']
dist_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# ============================================================================
# ۱. استخراج معیارهای اطلاعاتی برای همه توزیع‌ها
# ============================================================================
print("\n📊 استخراج معیارهای اطلاعاتی از Zarr...")

dataframes = {}

for dist in dist_names:
    aicc_var = f'{VAR}_{dist}_aicc'
    bic_var = f'{VAR}_{dist}_bic'
    loglik_var = f'{VAR}_{dist}_loglik'
    
    if aicc_var not in ds:
        print(f"   ⚠️ متغیر {aicc_var} وجود ندارد. از ادامه صرف‌نظر می‌شود.")
        continue
    
    aicc = ds[aicc_var].values.flatten()
    bic = ds[bic_var].values.flatten() if bic_var in ds else np.full_like(aicc, np.nan)
    loglik = ds[loglik_var].values.flatten() if loglik_var in ds else np.full_like(aicc, np.nan)
    
    valid_mask = np.isfinite(aicc)
    aicc_clean = aicc[valid_mask]
    bic_clean = bic[valid_mask]
    loglik_clean = loglik[valid_mask]
    
    print(f"   {dist_labels[dist_names.index(dist)]}: {len(aicc_clean)} مقدار معتبر از {len(aicc)}")
    
    dataframes[dist] = pd.DataFrame({
        'aicc': aicc_clean,
        'bic': bic_clean,
        'loglik': loglik_clean,
        'distribution': dist_labels[dist_names.index(dist)]
    })

# ترکیب همه داده‌ها
df_all = pd.concat(dataframes.values(), ignore_index=True)

# ============================================================================
# ۲. آماره‌های توصیفی
# ============================================================================
print("\n📊 آماره‌های توصیفی معیارهای اطلاعاتی:")
stats = df_all.groupby('distribution').agg({
    'aicc': ['count', 'mean', 'std', 'min', 'max'],
    'bic': ['count', 'mean', 'std', 'min', 'max'],
    'loglik': ['count', 'mean', 'std', 'min', 'max']
}).round(2)
print(stats)
stats.to_csv(f'gof_stats_{VAR}.csv')
print(f"   ✅ gof_stats_{VAR}.csv ذخیره شد.")

# ============================================================================
# ۳. درصد انتخاب هر توزیع (بر اساس best_dist)
# ============================================================================
print("\n📊 درصد انتخاب هر توزیع (بر اساس AICc):")

best_dist = ds[f'{VAR}_best_dist'].values.flatten()
valid_best = best_dist[best_dist >= 0]   # حذف -1 (Failed)
unique, counts = np.unique(valid_best, return_counts=True)
percentages = counts / len(valid_best) * 100

print("   کد توزیع → درصد:")
for code, pct in zip(unique, percentages):
    code_int = int(code)   # تبدیل به int برای استفاده به عنوان اندیس
    name = dist_labels[code_int] if code_int < len(dist_labels) else f"Unknown({code_int})"
    print(f"      {name}: {pct:.2f}%")

# ذخیره CSV
df_best = pd.DataFrame({
    'distribution_code': unique.astype(int),
    'distribution_name': [dist_labels[int(c)] if int(c) < len(dist_labels) else f"Unknown({int(c)})" for c in unique],
    'count': counts,
    'percentage': percentages
})
df_best.to_csv(f'best_distribution_percentages_{VAR}.csv', index=False)
print(f"   ✅ best_distribution_percentages_{VAR}.csv ذخیره شد.")

# ============================================================================
# ۴. نمودارهای مقایسه AICc
# ============================================================================
print("\n📊 رسم نمودارهای مقایسه...")

# باکس‌پلات (مقیاس لگاریتمی)
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_all, x='distribution', y='aicc', palette=dist_colors, ax=ax)
ax.set_yscale('log')
ax.set_xlabel('توزیع')
ax.set_ylabel('AICc (مقیاس لگاریتمی)')
ax.set_title(f'مقایسه توزیع AICc برای توزیع‌های مختلف ({VAR})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'boxplot_aicc_comparison_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ boxplot_aicc_comparison_{VAR}.png")

# هیستوگرام AICc برای هر توزیع
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, dist in enumerate(dist_names):
    if dist not in dataframes:
        axes[idx].set_visible(False)
        continue
    df_sub = dataframes[dist]
    ax = axes[idx]
    ax.hist(df_sub['aicc'], bins=50, alpha=0.7, color=dist_colors[idx], edgecolor='black')
    ax.set_xlabel('AICc')
    ax.set_ylabel('تعداد')
    ax.set_title(f'{dist_labels[idx]} (n={len(df_sub):,})')
    ax.grid(True, alpha=0.3)
for i in range(len(dist_names), len(axes)):
    axes[i].set_visible(False)
plt.tight_layout()
plt.savefig(f'histogram_aicc_{VAR}.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ histogram_aicc_{VAR}.png")

# ============================================================================
# ۵. بررسی مواردی که Bimodal بهتر از Normal عمل کرده است
# ============================================================================
print("\n📊 بررسی مواردی که Bimodal AICc کمتری نسبت به Normal دارد...")

if 'bimodal' in dataframes and 'normal' in dataframes:
    n_days = ds.dims['day']
    n_points = ds.dims['point']
    
    aicc_normal = ds[f'{VAR}_normal_aicc'].values
    aicc_bimodal = ds[f'{VAR}_bimodal_aicc'].values
    
    aicc_normal_flat = aicc_normal.flatten()
    aicc_bimodal_flat = aicc_bimodal.flatten()
    
    valid = np.isfinite(aicc_normal_flat) & np.isfinite(aicc_bimodal_flat)
    aicc_normal_valid = aicc_normal_flat[valid]
    aicc_bimodal_valid = aicc_bimodal_flat[valid]
    
    better_bimodal = aicc_bimodal_valid < aicc_normal_valid
    count_better = np.sum(better_bimodal)
    total = len(aicc_bimodal_valid)
    
    print(f"   تعداد کل مقایسه‌های معتبر: {total:,}")
    print(f"   تعداد مواردی که Bimodal AICc کمتری دارد: {count_better:,} ({count_better/total*100:.2f}%)")
    
    diff = aicc_bimodal_valid - aicc_normal_valid
    print(f"   میانگین تفاوت (Bimodal - Normal): {np.mean(diff):.2f}")
    print(f"   میانه تفاوت: {np.median(diff):.2f}")
    print(f"   انحراف معیار تفاوت: {np.std(diff):.2f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff, bins=100, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='مرز برابری')
    ax.set_xlabel('تفاوت AICc (Bimodal - Normal)')
    ax.set_ylabel('تعداد')
    ax.set_title(f'توزیع تفاوت AICc بین Bimodal و Normal ({VAR})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'diff_aicc_bimodal_normal_{VAR}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ diff_aicc_bimodal_normal_{VAR}.png")

# ============================================================================
# ۶. جمع‌بندی
# ============================================================================
print("\n" + "=" * 80)
print("✅ تحلیل کامل شد.")
print("📁 فایل‌های خروجی:")
for f in [
    f'gof_stats_{VAR}.csv',
    f'best_distribution_percentages_{VAR}.csv',
    f'boxplot_aicc_comparison_{VAR}.png',
    f'histogram_aicc_{VAR}.png',
    f'diff_aicc_bimodal_normal_{VAR}.png'
]:
    print(f"   - {f}")
print("=" * 80)

ds.close()