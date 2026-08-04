# -*- coding: utf-8 -*-
"""
numerical_engine/analyze_station.py
================================================================================
تحلیل یک ایستگاه: محاسبه آماره‌ها، برازش ۵ توزیع و تولید دیکشنری کامل خروجی
همه کلیدهای تعریف‌شده در zarr_schema.VAR_NAMES تولید می‌شوند.
================================================================================
"""

import numpy as np
from constants import VARS, N_DAYS, MIN_VALID_VALUES
from numerical_engine.distributions import fit_distribution
from numerical_engine.window_engine import extract_window_values


def analyze_station(station_data, year_list, window_table, var_idx):
    """
    پارامترها:
        station_data: آرایه (N_YEARS, N_DAYS, N_VARS) یا (N_YEARS, N_DAYS)
        year_list: لیست سال‌ها
        window_table: جدول پنجره‌ها (هر روز شامل لیست روزهای پنجره)
        var_idx: اندیس متغیر مورد نظر (0= tmin, 1= tmean, 2= tmax)

    بازگشت:
        دیکشنری با کلیدهای منطبق بر VAR_NAMES
    """
    # --------------------------------------------
    # ۰. آماده‌سازی داده
    # --------------------------------------------
    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)

    # تبدیل به شکل استاندارد (N_YEARS, N_DAYS, N_VARS)
    if station_data.ndim == 1:
        # اگر یک‌بعدی است، فرض می‌کنیم به ترتیب (سال, روز, متغیر) پشت سر هم چیده شده
        n_years = len(year_list)
        n_days = N_DAYS
        n_vars = len(VARS)
        if station_data.size == n_years * n_days * n_vars:
            station_data = station_data.reshape(n_years, n_days, n_vars)
        else:
            # داده نامعتبر
            return _empty_result(var_idx)
    elif station_data.ndim == 2:
        # اگر دو‌بعدی است، فرض می‌کنیم (N_YEARS*N_DAYS, N_VARS) یا (N_YEARS, N_DAYS)
        if station_data.shape[0] == len(year_list) and station_data.shape[1] == N_DAYS:
            # (N_YEARS, N_DAYS) -> با یک متغیر
            station_data = station_data.reshape(len(year_list), N_DAYS, 1)
        elif station_data.shape[1] == len(VARS):
            # (N_YEARS*N_DAYS, N_VARS) -> بازآرایی
            n_years = len(year_list)
            n_days = N_DAYS
            station_data = station_data.reshape(n_years, n_days, -1)
        else:
            return _empty_result(var_idx)

    # اطمینان از بعد سوم (متغیرها)
    if station_data.ndim == 2:
        station_data = station_data.reshape(station_data.shape[0], station_data.shape[1], 1)

    N_YEARS, N_DAYS, n_vars = station_data.shape
    var_name = VARS[var_idx] if var_idx < len(VARS) else 'tmean'

    # استخراج داده‌های مربوط به این متغیر
    data_3d = station_data[:, :, var_idx]  # (N_YEARS, N_DAYS)

    # --------------------------------------------
    # ۱. حلقه روی روزها و برازش
    # --------------------------------------------
    # آرایه‌های ذخیره نتایج روزانه
    n_days = N_DAYS
    results = {
        'count': np.zeros(n_days, dtype=np.int32),
        'mean': np.full(n_days, np.nan, dtype=np.float32),
        'std': np.full(n_days, np.nan, dtype=np.float32),
        'median': np.full(n_days, np.nan, dtype=np.float32),
        'min': np.full(n_days, np.nan, dtype=np.float32),
        'max': np.full(n_days, np.nan, dtype=np.float32),
        'skewness': np.full(n_days, np.nan, dtype=np.float32),
        'best_dist': np.full(n_days, -1, dtype=np.int32),
    }

    # دیکشنری برای پارامترهای هر توزیع (۵ توزیع)
    dist_names = ['normal', 'skew', 'gev', 'bimodal', 'pearson']
    for dist in dist_names:
        for p in range(1, 6):
            results[f'{dist}_p{p}'] = np.full(n_days, np.nan, dtype=np.float32)
        results[f'{dist}_aicc'] = np.full(n_days, np.nan, dtype=np.float32)
        results[f'{dist}_bic'] = np.full(n_days, np.nan, dtype=np.float32)
        results[f'{dist}_loglik'] = np.full(n_days, np.nan, dtype=np.float32)

    # --------------------------------------------
    # ۲. پردازش هر روز
    # --------------------------------------------
    for day_idx in range(n_days):
        # استخراج پنجره
        window_days = window_table[day_idx]  # لیست روزها (0-based)
        # جمع‌آوری مقادیر برای همه سال‌ها
        values = []
        for year_idx in range(N_YEARS):
            for d in window_days:
                val = data_3d[year_idx, d]
                if not np.isnan(val):
                    values.append(val)

        values = np.array(values, dtype=np.float64)
        n_valid = len(values)

        # ذخیره آماره‌ها
        results['count'][day_idx] = n_valid
        if n_valid >= MIN_VALID_VALUES:
            results['mean'][day_idx] = np.mean(values)
            results['std'][day_idx] = np.std(values)
            results['median'][day_idx] = np.median(values)
            results['min'][day_idx] = np.min(values)
            results['max'][day_idx] = np.max(values)
            results['skewness'][day_idx] = np.mean(((values - results['mean'][day_idx]) / results['std'][day_idx]) ** 3)

            # برازش همه توزیع‌ها
            fit_result = fit_distribution(values)  # آرایه ۳۳ عنصری
            if fit_result is not None and not np.isnan(fit_result[0]):
                best_code = int(fit_result[0])
                results['best_dist'][day_idx] = best_code

                # استخراج پارامترها برای هر توزیع
                # ترتیب خروجی fit_distribution:
                # [best, normal_p1, normal_p2, normal_loglik, normal_aicc, normal_bic, normal_k,
                #  skew_p1, skew_p2, skew_p3, skew_loglik, skew_aicc, skew_bic, skew_k,
                #  gev_p1, gev_p2, gev_p3, gev_loglik, gev_aicc, gev_bic, gev_k,
                #  bimodal_p1, bimodal_p2, bimodal_p3, bimodal_p4, bimodal_p5, bimodal_loglik, bimodal_aicc, bimodal_bic, bimodal_k,
                #  pearson_p1, pearson_p2, pearson_p3, pearson_loglik, pearson_aicc, pearson_bic, pearson_k]
                # (تعداد ۳۳ عنصر)

                # نرمال (کد ۰)
                results['normal_p1'][day_idx] = fit_result[1]
                results['normal_p2'][day_idx] = fit_result[2]
                results['normal_p3'][day_idx] = np.nan
                results['normal_p4'][day_idx] = np.nan
                results['normal_p5'][day_idx] = np.nan
                results['normal_aicc'][day_idx] = fit_result[4]
                results['normal_bic'][day_idx] = fit_result[5]
                results['normal_loglik'][day_idx] = fit_result[3]

                # Skew-Normal (کد ۱)
                results['skew_p1'][day_idx] = fit_result[7]
                results['skew_p2'][day_idx] = fit_result[8]
                results['skew_p3'][day_idx] = fit_result[9]
                results['skew_p4'][day_idx] = np.nan
                results['skew_p5'][day_idx] = np.nan
                results['skew_aicc'][day_idx] = fit_result[11]
                results['skew_bic'][day_idx] = fit_result[12]
                results['skew_loglik'][day_idx] = fit_result[10]

                # GEV (کد ۲)
                results['gev_p1'][day_idx] = fit_result[14]
                results['gev_p2'][day_idx] = fit_result[15]
                results['gev_p3'][day_idx] = fit_result[16]
                results['gev_p4'][day_idx] = np.nan
                results['gev_p5'][day_idx] = np.nan
                results['gev_aicc'][day_idx] = fit_result[18]
                results['gev_bic'][day_idx] = fit_result[19]
                results['gev_loglik'][day_idx] = fit_result[17]

                # Bimodal (کد ۳)
                results['bimodal_p1'][day_idx] = fit_result[21]
                results['bimodal_p2'][day_idx] = fit_result[22]
                results['bimodal_p3'][day_idx] = fit_result[23]
                results['bimodal_p4'][day_idx] = fit_result[24]
                results['bimodal_p5'][day_idx] = fit_result[25]
                results['bimodal_aicc'][day_idx] = fit_result[27]
                results['bimodal_bic'][day_idx] = fit_result[28]
                results['bimodal_loglik'][day_idx] = fit_result[26]

                # Pearson (کد ۴)
                results['pearson_p1'][day_idx] = fit_result[30]
                results['pearson_p2'][day_idx] = fit_result[31]
                results['pearson_p3'][day_idx] = fit_result[32]
                results['pearson_p4'][day_idx] = np.nan
                results['pearson_p5'][day_idx] = np.nan
                results['pearson_aicc'][day_idx] = fit_result[33] if len(fit_result) > 33 else np.nan
                results['pearson_bic'][day_idx] = fit_result[34] if len(fit_result) > 34 else np.nan
                results['pearson_loglik'][day_idx] = fit_result[35] if len(fit_result) > 35 else np.nan

    # --------------------------------------------
    # ۳. ساخت دیکشنری نهایی با کلیدهای استاندارد
    # --------------------------------------------
    final = {}
    # آماره‌ها
    for stat in ['count', 'mean', 'std', 'median', 'min', 'max', 'skewness']:
        final[f'{var_name}_{stat}'] = results[stat]

    # پارامترها و معیارهای هر توزیع
    for dist in dist_names:
        for p in range(1, 6):
            final[f'{var_name}_{dist}_p{p}'] = results[f'{dist}_p{p}']
        final[f'{var_name}_{dist}_aicc'] = results[f'{dist}_aicc']
        final[f'{var_name}_{dist}_bic'] = results[f'{dist}_bic']
        final[f'{var_name}_{dist}_loglik'] = results[f'{dist}_loglik']

    # بهترین توزیع
    final[f'{var_name}_best_dist'] = results['best_dist']

    return final


def _empty_result(var_idx):
    """برگرداندن دیکشنری خالی (همه NaN) برای زمانی که داده معتبری وجود ندارد."""
    var_name = VARS[var_idx] if var_idx < len(VARS) else 'tmean'
    final = {}
    n_days = N_DAYS

    # آماره‌ها
    for stat in ['count', 'mean', 'std', 'median', 'min', 'max', 'skewness']:
        final[f'{var_name}_{stat}'] = np.full(n_days, np.nan, dtype=np.float32)

    dist_names = ['normal', 'skew', 'gev', 'bimodal', 'pearson']
    for dist in dist_names:
        for p in range(1, 6):
            final[f'{var_name}_{dist}_p{p}'] = np.full(n_days, np.nan, dtype=np.float32)
        final[f'{var_name}_{dist}_aicc'] = np.full(n_days, np.nan, dtype=np.float32)
        final[f'{var_name}_{dist}_bic'] = np.full(n_days, np.nan, dtype=np.float32)
        final[f'{var_name}_{dist}_loglik'] = np.full(n_days, np.nan, dtype=np.float32)

    final[f'{var_name}_best_dist'] = np.full(n_days, -1, dtype=np.int32)
    return final