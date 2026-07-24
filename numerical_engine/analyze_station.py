#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/analyze_station.py
================================================================================
تحلیل یک ایستگاه کامل.
ورودی: (N_YEARS, N_DAYS, N_VARS) → خروجی: ۳۳ آرایه (N_DAYS,)
کاملاً مستقل از فایل و I/O.
================================================================================
ورژن: 2.0 - نهایی
"""

import numpy as np
from numerical_engine.window_engine import extract_window_values_fast
from numerical_engine.distributions import fit_distribution
from zarr_schema import VAR_NAMES
from constants import N_DAYS, VAR_INDEX_FOR_FIT

def analyze_station(station_data, window_table, var_idx=VAR_INDEX_FOR_FIT):
    result = {}
    for name in VAR_NAMES:
        if name == "best_dist":
            result[name] = np.full(N_DAYS, -1, dtype=np.int32)
        elif name == "count":
            result[name] = np.zeros(N_DAYS, dtype=np.int32)
        else:
            result[name] = np.full(N_DAYS, np.nan, dtype=np.float32)

    windows = extract_window_values_fast(station_data, window_table, var_idx)

    for doy_idx, values in enumerate(windows):
        if values is None:
            continue
        res = fit_distribution(values)
        if res is None or np.isnan(res[0]):
            continue

        best_code = int(res[0])
        result["best_dist"][doy_idx] = best_code
        result["normal_p1"][doy_idx] = res[1]
        result["normal_p2"][doy_idx] = res[2]
        result["normal_loglik"][doy_idx] = res[3]
        result["normal_aicc"][doy_idx] = res[4]
        result["normal_bic"][doy_idx] = res[5]
        result["skew_p1"][doy_idx] = res[7]
        result["skew_p2"][doy_idx] = res[8]
        result["skew_p3"][doy_idx] = res[9]
        result["skew_loglik"][doy_idx] = res[10]
        result["skew_aicc"][doy_idx] = res[11]
        result["skew_bic"][doy_idx] = res[12]
        result["gev_p1"][doy_idx] = res[14]
        result["gev_p2"][doy_idx] = res[15]
        result["gev_p3"][doy_idx] = res[16]
        result["gev_loglik"][doy_idx] = res[17]
        result["gev_aicc"][doy_idx] = res[18]
        result["gev_bic"][doy_idx] = res[19]
        result["pearson_p1"][doy_idx] = res[21]
        result["pearson_p2"][doy_idx] = res[22]
        result["pearson_p3"][doy_idx] = res[23]
        result["pearson_loglik"][doy_idx] = res[24]
        result["pearson_aicc"][doy_idx] = res[25]
        result["pearson_bic"][doy_idx] = res[26]
        result["mean"][doy_idx] = res[28]
        result["std"][doy_idx] = res[29]
        result["skewness"][doy_idx] = res[30]
        result["median"][doy_idx] = res[31]
        result["count"][doy_idx] = int(res[32])

    return result
