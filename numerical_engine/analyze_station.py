# -*- coding: utf-8 -*-
"""
analyze_station.py - تحلیل آماری یک ایستگاه (برای همه متغیرها)
"""

import numpy as np
from numerical_engine.distributions import fit_distributions, select_best_distribution
from numerical_engine.window_engine import compute_windows

# نگاشت نام توزیع به عدد
DIST_MAP = {"normal": 0, "pearson": 1, "skewnormal": 2, "bimodal": 3}

def analyze_station(station_data, year_list, window_table, var_idx=None):
    """
    تحلیل آماری داده‌های یک ایستگاه برای همه متغیرها

    Parameters
    ----------
    station_data : np.ndarray
        shape (N_YEARS, N_DAYS, n_vars)
    year_list : list
        لیست سال‌ها
    window_table : list or dict
        جدول پنجره‌ها
    var_idx : int (ignored)
        برای سازگاری نگه داشته شده است (همه متغیرها پردازش می‌شوند)

    Returns
    -------
    results : dict
        کلیدها: f"{var_name}_{stat_name}" و f"{var_name}_{dist_name}_p{i}"
    """
    # ============================================================
    # ۰. بررسی ورودی
    # ============================================================
    if not isinstance(station_data, np.ndarray):
        station_data = np.array(station_data)

    # اطمینان از ابعاد صحیح
    if station_data.ndim == 0:
        station_data = station_data.reshape(1, 1, 1)
    elif station_data.ndim == 1:
        N_YEARS = len(year_list)
        N_DAYS = 366
        n_vars = station_data.shape[0] // (N_YEARS * N_DAYS)
        if n_vars > 0 and station_data.size == N_YEARS * N_DAYS * n_vars:
            station_data = station_data.reshape(N_YEARS, N_DAYS, n_vars)
        else:
            station_data = np.full((len(year_list), 366, 3), np.nan, dtype=np.float32)
    elif station_data.ndim == 2:
        if station_data.shape == (len(year_list), 366):
            station_data = station_data.reshape(len(year_list), 366, 1)
        elif station_data.shape[0] == len(year_list) * 366:
            station_data = station_data.reshape(len(year_list), 366, -1)

    if station_data.size == 0 or np.all(np.isnan(station_data)):
        N_YEARS = len(year_list)
        N_DAYS = 366
        n_vars = station_data.shape[-1] if station_data.ndim > 0 else 3
        return _empty_result(N_YEARS, N_DAYS, n_vars)

    N_YEARS, N_DAYS, n_vars = station_data.shape
    results = {}

    # لیست نام توزیع‌ها
    dist_names = ["normal", "pearson", "skewnormal", "bimodal"]

    for v in range(n_vars):
        var_name = ["tmax", "tmean", "tmin"][v] if v < 3 else f"var_{v}"
        var_data = station_data[:, :, v]  # shape: (N_YEARS, N_DAYS)

        # ============================================================
        # آمار پایه برای هر روز
        # ============================================================
        daily_stats = {
            "count": np.sum(~np.isnan(var_data), axis=0),
            "mean": np.nanmean(var_data, axis=0),
            "std": np.nanstd(var_data, axis=0),
            "median": np.nanmedian(var_data, axis=0),
            "min": np.nanmin(var_data, axis=0),
            "max": np.nanmax(var_data, axis=0),
            "skewness": _skewness(var_data),
        }

        for stat_name, stat_data in daily_stats.items():
            results[f"{var_name}_{stat_name}"] = stat_data.astype(np.float32)

        # ============================================================
        # برازش توزیع‌ها برای این متغیر
        # ============================================================
        # مقداردهی اولیه برای نتایج توزیع‌ها (با NaN)
        dist_results = {}
        for dist in dist_names:
            for p in range(1, 6):
                dist_results[f"{var_name}_{dist}_p{p}"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        dist_results[f"{var_name}_best_dist"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        dist_results[f"{var_name}_aic"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        dist_results[f"{var_name}_bic"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        dist_results[f"{var_name}_loglik"] = np.full(N_DAYS, np.nan, dtype=np.float32)

        # محاسبه پنجره‌ها و برازش برای هر روز
        windows = compute_windows(var_data, window_table, year_list)
        for day_idx in range(N_DAYS):
            day_data = windows[day_idx]
            if len(day_data) > 10:
                fits = fit_distributions(day_data)
                best = select_best_distribution(fits)
                if best is not None and best in DIST_MAP:
                    dist_results[f"{var_name}_best_dist"][day_idx] = float(DIST_MAP[best])
                else:
                    dist_results[f"{var_name}_best_dist"][day_idx] = np.nan
                if best is not None and best in fits:
                    dist_results[f"{var_name}_aic"][day_idx] = fits[best].get("aic", np.nan)
                    dist_results[f"{var_name}_bic"][day_idx] = fits[best].get("bic", np.nan)
                    dist_results[f"{var_name}_loglik"][day_idx] = fits[best].get("loglik", np.nan)
                # ذخیره پارامترهای توزیع‌ها
                for dist_name, params in fits.items():
                    for p_idx, p_val in enumerate(params.values()):
                        if p_idx < 5:
                            key = f"{var_name}_{dist_name}_p{p_idx+1}"
                            if key in dist_results:
                                dist_results[key][day_idx] = p_val

        # اضافه کردن dist_results به results
        for key, value in dist_results.items():
            results[key] = value

    return results


def _skewness(data):
    """محاسبه چولگی در طول سال‌ها برای هر روز"""
    N_YEARS, N_DAYS = data.shape
    skew = np.full(N_DAYS, np.nan, dtype=np.float32)
    for d in range(N_DAYS):
        day_data = data[:, d]
        day_data = day_data[~np.isnan(day_data)]
        if len(day_data) > 2:
            try:
                from scipy import stats
                skew[d] = stats.skew(day_data)
            except:
                pass
    return skew


def _empty_result(N_YEARS, N_DAYS, n_vars):
    """برگرداندن نتایج خالی با NaN"""
    results = {}
    var_names = ["tmax", "tmean", "tmin"]
    stat_names = ["count", "mean", "std", "median", "min", "max", "skewness"]

    for var in var_names[:n_vars]:
        for stat in stat_names:
            results[f"{var}_{stat}"] = np.full(N_DAYS, np.nan, dtype=np.float32)

        dist_names = ["normal", "pearson", "skewnormal", "bimodal"]
        for dist in dist_names:
            for p in range(1, 6):
                results[f"{var}_{dist}_p{p}"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        results[f"{var}_best_dist"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        results[f"{var}_aic"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        results[f"{var}_bic"] = np.full(N_DAYS, np.nan, dtype=np.float32)
        results[f"{var}_loglik"] = np.full(N_DAYS, np.nan, dtype=np.float32)

    return results