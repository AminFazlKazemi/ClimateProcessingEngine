#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_distributions.py - بررسی کامل توزیع‌ها
"""

import os
import sys
import numpy as np
import re
from pathlib import Path

print("=" * 70)
print("🔍 بررسی توزیع‌های موجود در پروژه")
print("=" * 70)

# ============================================================================
# ۱. بررسی فایل distributions.py
# ============================================================================
print("\n📄 ۱. بررسی distributions.py ...")

dist_path = Path("numerical_engine/distributions.py")
if not dist_path.exists():
    print("   ❌ distributions.py یافت نشد!")
    sys.exit(1)

with open(dist_path, 'r', encoding='utf-8') as f:
    content = f.read()

distributions = {
    'Normal': 'fit_normal_full_numba',
    'Skew-Normal': 'fit_skewnorm_full_numba',
    'GEV': 'fit_gev_full_numba',
    'Bimodal': 'fit_bimodal_full_numba',
    'Pearson': 'fit_pearson3_full_numba',
}

all_found = True
for name, func in distributions.items():
    if func in content:
        print(f"   ✅ {name}: {func} موجود است")
    else:
        print(f"   ❌ {name}: {func} یافت نشد!")
        all_found = False

# بررسی fastmath
if "fastmath=True" in content:
    print("   ✅ fastmath=True در توابع Numba فعال است")
else:
    print("   ⚠️ fastmath=True یافت نشد (اختیاری)")

# ============================================================================
# ۲. تست برازش با داده‌های نمونه
# ============================================================================
print("\n🧪 ۲. تست برازش با داده‌های نمونه ...")

try:
    from numerical_engine.distributions import fit_distribution
    
    # داده‌های تست
    data = np.random.normal(0, 1, 200)
    result = fit_distribution(data)
    
    if result is not None and not np.isnan(result[0]):
        best = int(result[0])
        names = {0: 'Normal', 1: 'Skew-Normal', 2: 'GEV', 3: 'Bimodal', 4: 'Pearson'}
        print(f"   ✅ برازش موفق: بهترین توزیع = {names.get(best, 'نامشخص')}")
        print(f"      Normal AICc: {result[4]:.2f}")
        print(f"      Skew AICc:   {result[11]:.2f}")
        print(f"      GEV AICc:    {result[18]:.2f}")
        print(f"      Bimodal AICc:{result[27]:.2f}")
        print(f"      Pearson AICc:{result[30]:.2f}")
    else:
        print("   ❌ برازش ناموفق!")
        all_found = False
except Exception as e:
    print(f"   ❌ خطا در برازش: {e}")
    all_found = False

# ============================================================================
# ۳. بررسی خروجی Zarr (در صورت وجود)
# ============================================================================
print("\n📊 ۳. بررسی خروجی Zarr ...")

zarr_path = "I:/climatology_366_rolling/climatology_stationwise_final.zarr"
if os.path.exists(zarr_path):
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path, consolidated=False)
        
        dist_vars = {
            'Normal': ['normal_p1', 'normal_p2', 'normal_loglik', 'normal_aicc', 'normal_bic'],
            'Skew': ['skew_p1', 'skew_p2', 'skew_p3', 'skew_loglik', 'skew_aicc', 'skew_bic'],
            'GEV': ['gev_p1', 'gev_p2', 'gev_p3', 'gev_loglik', 'gev_aicc', 'gev_bic'],
            'Bimodal': ['bimodal_p1', 'bimodal_p2', 'bimodal_p3', 'bimodal_p4', 'bimodal_p5',
                        'bimodal_loglik', 'bimodal_aicc', 'bimodal_bic'],
            'Pearson': ['pearson_p1', 'pearson_p2', 'pearson_p3', 'pearson_loglik', 'pearson_aicc', 'pearson_bic'],
        }
        
        for name, vars_list in dist_vars.items():
            exists = all(v in ds.data_vars for v in vars_list)
            if exists:
                print(f"   ✅ {name}: همه متغیرها موجود هستند")
            else:
                missing = [v for v in vars_list if v not in ds.data_vars]
                print(f"   ❌ {name}: متغیرهای زیر موجود نیستند: {missing}")
                all_found = False
        
        ds.close()
    except Exception as e:
        print(f"   ⚠️ خطا در خواندن Zarr: {e}")
        print("   (این خطا طبیعی است اگر Zarr هنوز ساخته نشده باشد)")
else:
    print("   ⚠️ فایل Zarr خروجی هنوز وجود ندارد (اجرا نشده)")

# ============================================================================
# ۴. نتیجه‌گیری
# ============================================================================
print("\n" + "=" * 70)
if all_found:
    print("✅ همه توزیع‌ها سالم هستند! هیچکدام حذف نشده‌اند.")
else:
    print("⚠️ برخی از توزیع‌ها مشکل دارند. لطفاً بررسی کنید.")
print("=" * 70)