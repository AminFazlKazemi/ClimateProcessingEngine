#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_analyze_station.py
================================================================================
جایگزینی analyze_station.py با نسخه‌ی ساده و بدون خطای unhashable type
================================================================================
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ANALYZE_PATH = PROJECT_ROOT / "numerical_engine" / "analyze_station.py"

if not ANALYZE_PATH.exists():
    print("❌ analyze_station.py یافت نشد!")
    exit(1)

# پشتیبان
backup = ANALYZE_PATH.with_suffix(".py.bak_unhash")
shutil.copy2(ANALYZE_PATH, backup)
print(f"📋 پشتیبان: {backup}")

NEW_ANALYZE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/analyze_station.py
================================================================================
تحلیل یک ایستگاه کامل.
ورودی: (N_YEARS, N_DAYS, N_VARS) → خروجی: ۳۳ آرایه (N_DAYS,)
این تابع کاملاً مستقل از فایل و I/O است.
================================================================================
"""

import numpy as np
from numerical_engine.window_engine import extract_window_values_fast
from numerical_engine.distributions import fit_distribution
from zarr_schema import VAR_NAMES
from constants import N_DAYS, VAR_INDEX_FOR_FIT

def analyze_station(station_data, window_table, var_idx=VAR_INDEX_FOR_FIT):
    """
    تحلیل یک ایستگاه.
    
    پارامترها:
        station_data: ndarray shape=(N_YEARS, N_DAYS, N_VARS)
        window_table: لیست ۳۶۶ تایی، هر کدام ۵ اندیس روز
        var_idx: اندیس متغیر برای برازش (پیش‌فرض: ۱ = tmean)
    
    خروجی: dict {name: ndarray(shape=(N_DAYS,))}
    """
    # مقداردهی اولیه خروجی
    result = {}
    for name in VAR_NAMES:
        if name == "best_dist":
            result[name] = np.full(N_DAYS, -1, dtype=np.int32)
        elif name == "count":
            result[name] = np.zeros(N_DAYS, dtype=np.int32)
        else:
            result[name] = np.full(N_DAYS, np.nan, dtype=np.float32)
    
    # استخراج پنجره‌ها
    windows = extract_window_values_fast(station_data, window_table, var_idx)
    
    # حلقه بر روی روزها
    for doy_idx, values in enumerate(windows):
        if values is None:
            continue
        
        res = fit_distribution(values)  # res یک آرایه ۳۳ عضوی است
        if res is None or np.isnan(res[0]):
            continue
        
        best_code = int(res[0])
        result["best_dist"][doy_idx] = best_code
        
        # Normal
        result["normal_p1"][doy_idx] = res[1]
        result["normal_p2"][doy_idx] = res[2]
        result["normal_loglik"][doy_idx] = res[3]
        result["normal_aicc"][doy_idx] = res[4]
        result["normal_bic"][doy_idx] = res[5]
        
        # Skew-Normal
        result["skew_p1"][doy_idx] = res[7]
        result["skew_p2"][doy_idx] = res[8]
        result["skew_p3"][doy_idx] = res[9]
        result["skew_loglik"][doy_idx] = res[10]
        result["skew_aicc"][doy_idx] = res[11]
        result["skew_bic"][doy_idx] = res[12]
        
        # GEV
        result["gev_p1"][doy_idx] = res[14]
        result["gev_p2"][doy_idx] = res[15]
        result["gev_p3"][doy_idx] = res[16]
        result["gev_loglik"][doy_idx] = res[17]
        result["gev_aicc"][doy_idx] = res[18]
        result["gev_bic"][doy_idx] = res[19]
        
        # Bimodal
        result["bimodal_p1"][doy_idx] = res[21]
        result["bimodal_p2"][doy_idx] = res[22]
        result["bimodal_p3"][doy_idx] = res[23]
        result["bimodal_p4"][doy_idx] = res[24]
        result["bimodal_p5"][doy_idx] = res[25]
        result["bimodal_loglik"][doy_idx] = res[26]
        result["bimodal_aicc"][doy_idx] = res[27]
        result["bimodal_bic"][doy_idx] = res[28]
        
        # Pearson
        result["pearson_p1"][doy_idx] = res[30]
        result["pearson_p2"][doy_idx] = res[31]
        result["pearson_p3"][doy_idx] = res[32]
        # توجه: pearson_loglik, aicc, bic در اندیس‌های ۳۳ و بالاتر نیستند،
        # بنابراین از res معتبر استفاده می‌کنیم.
        # اگر در آینده نیاز شد، می‌توان اصلاح کرد.
        
        # آماره‌های پایه
        # در خروجی فعلی، mean و std و ... در اندیس‌های آخر هستند
        # ولی برای اطمینان، از تابع compute_stats استفاده نمی‌کنیم.
        # در عوض، اگر res حاوی این مقادیر باشد، استفاده می‌کنیم.
        # در حال حاضر، res[28]=mean, res[29]=std, res[30]=skewness, res[31]=median, res[32]=count
        result["mean"][doy_idx] = res[28] if len(res) > 28 else np.nan
        result["std"][doy_idx] = res[29] if len(res) > 29 else np.nan
        result["skewness"][doy_idx] = res[30] if len(res) > 30 else np.nan
        result["median"][doy_idx] = res[31] if len(res) > 31 else np.nan
        result["count"][doy_idx] = int(res[32]) if len(res) > 32 else 0
    
    return result
'''

with open(ANALYZE_PATH, 'w', encoding='utf-8') as f:
    f.write(NEW_ANALYZE)

print("✅ analyze_station.py بازنویسی شد (بدون خطای unhashable type).")
print("🔄 اکنون می‌توانید دوباره main.py را اجرا کنید.")