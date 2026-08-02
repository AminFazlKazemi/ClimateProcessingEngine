#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/analyze_station.py
================================================================================
تحلیل یک ایستگاه کامل.
ورودی: (N_YEARS, N_DAYS, N_VARS) → خروجی: دیکشنری با کلیدهای دقیق VAR_NAMES
نسخه ۴.۰ – با ثبت outlierها
================================================================================
"""

import numpy as np
from numerical_engine.window_engine import (
    extract_window_values_fast,
    extract_window_values_raw,
    MIN_TEMP_RAW,
    MAX_TEMP_RAW
)
from numerical_engine.distributions import fit_distribution
from zarr_schema import VAR_NAMES, DISTRIBUTIONS
from constants import N_DAYS, VARS, MIN_VALID_VALUES
from monitoring.outlier_logger import log_outlier

# نگاشت کد توزیع به نام
DIST_CODE_TO_NAME = {code: info["name"].lower() for code, info in DISTRIBUTIONS.items()}

def analyze_station(station_data, year_list, window_table, var_idx, station_idx):
    """
    تحلیل یک ایستگاه برای همه متغیرها.
    """
    # مقداردهی اولیه خروجی با کلیدهای VAR_NAMES
    result = {}
    for name in VAR_NAMES:
        if 'best_dist' in name or 'count' in name:
            result[name] = np.full(N_DAYS, -1 if 'best_dist' in name else 0, dtype=np.int32)
        else:
            result[name] = np.full(N_DAYS, np.nan, dtype=np.float32)

    try:
        # پردازش هر متغیر
        for v_idx, var_name in enumerate(VARS):
            # داده‌های خام (بدون فیلتر) برای شناسایی outlier
            raw_windows = extract_window_values_raw(station_data, window_table, v_idx)
            # داده‌های پاک‌شده برای برازش
            clean_windows = extract_window_values_fast(station_data, window_table, v_idx)

            for doy_idx, (raw_vals, clean_vals) in enumerate(zip(raw_windows, clean_windows)):
                # ثبت outlierها از داده‌های خام
                if raw_vals is not None:
                    for val in raw_vals:
                        if not np.isnan(val) and (val < MIN_TEMP_RAW or val > MAX_TEMP_RAW):
                            log_outlier(station_idx, doy_idx, val, var_name)

                # ادامه برازش با داده‌های پاک‌شده
                if clean_vals is None or len(clean_vals) < MIN_VALID_VALUES:
                    continue

                res = fit_distribution(clean_vals)  # آرایه ۳۸ عضوی
                if res is None or np.isnan(res[0]):
                    continue

                best_code = int(res[0])
                result[f"{var_name}_best_dist"][doy_idx] = best_code

                # آماره‌های پایه (اندیس‌های ۳۳–۳۷)
                result[f"{var_name}_mean"][doy_idx] = res[33]
                result[f"{var_name}_std"][doy_idx] = res[34]
                result[f"{var_name}_skewness"][doy_idx] = res[35]
                result[f"{var_name}_median"][doy_idx] = res[36]
                result[f"{var_name}_count"][doy_idx] = int(res[37]) if not np.isnan(res[37]) else 0

                # پارامترها و معیارهای اطلاعاتی برای هر توزیع
                offsets = {
                    'normal': 1,
                    'skew': 7,
                    'gev': 14,
                    'bimodal': 21,
                    'pearson': 30,
                }
                info_indices = {
                    'normal': (3, 4, 5),
                    'skew': (10, 11, 12),
                    'gev': (17, 18, 19),
                    'bimodal': (26, 27, 28),
                    'pearson': (None, None, None),
                }
                n_params = {
                    'normal': 2,
                    'skew': 3,
                    'gev': 3,
                    'bimodal': 5,
                    'pearson': 3,
                }

                for dist_name, offset in offsets.items():
                    prefix = f"{var_name}_{dist_name}"
                    for i in range(n_params[dist_name]):
                        idx = offset + i
                        if idx < len(res) and not np.isnan(res[idx]):
                            result[f"{prefix}_p{i+1}"][doy_idx] = res[idx]
                    loglik_idx, aicc_idx, bic_idx = info_indices[dist_name]
                    if loglik_idx is not None and loglik_idx < len(res):
                        result[f"{prefix}_loglik"][doy_idx] = res[loglik_idx]
                        result[f"{prefix}_aicc"][doy_idx] = res[aicc_idx]
                        result[f"{prefix}_bic"][doy_idx] = res[bic_idx]

    except Exception as e:
        print(f"⚠️ analyze_station error: {e}")

    return result