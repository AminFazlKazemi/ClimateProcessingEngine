# -*- coding: utf-8 -*-
"""
apply_patches.py - اعمال اصلاحات با جستجوی پیشرفته
"""

import os
import re
from pathlib import Path

# مسیر پایه: پوشه‌ای که اسکریپت در آن قرار دارد
BASE_DIR = Path(__file__).parent

def find_file(filename, search_dirs=None):
    """پیدا کردن فایل در دایرکتوری‌های مشخص"""
    if search_dirs is None:
        search_dirs = [BASE_DIR]
    if isinstance(search_dirs, (str, Path)):
        search_dirs = [search_dirs]

    for base in search_dirs:
        for root, dirs, files in os.walk(base):
            if filename in files:
                return Path(root) / filename
    return None

def patch_analyze_station():
    """اضافه کردن بررسی ابعاد به analyze_station.py"""
    # جستجو در دایرکتوری‌های اصلی
    file_path = find_file("analyze_station.py", [
        BASE_DIR / "numerical_engine",
        BASE_DIR / "numerical_engine" / "statistics",
        BASE_DIR
    ])

    if file_path is None:
        print("⚠️ فایل analyze_station.py یافت نشد. لطفاً مسیر را بررسی کنید.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # اگر قبلاً اصلاح شده، خروج
    if "if not isinstance(station_data, np.ndarray):" in content:
        print("✅ analyze_station.py قبلاً اصلاح شده است.")
        return

    # پیدا کردن تابع analyze_station
    pattern = r'(def analyze_station\(station_data, year_list, window_table, var_idx\):.*?)(?=\n    # ============================================================|\n    # ادامه کد|\n    if np\.all\(np\.isnan)'
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("⚠️ تابع analyze_station در analyze_station.py پیدا نشد.")
        return

    func_start = match.group(1)
    # کد بررسی ابعاد
    dimension_check = '''
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

    N_YEARS, N_DAYS, n_vars = station_data.shape'''

    # درج کد بعد از def و docstring
    doc_pattern = r'("""[\s\S]*?""")'
    doc_match = re.search(doc_pattern, func_start)
    if doc_match:
        new_func_start = func_start.replace(doc_match.group(0), doc_match.group(0) + '\n' + dimension_check)
    else:
        new_func_start = func_start + '\n' + dimension_check

    content = content.replace(func_start, new_func_start)

    # اضافه کردن تابع _empty_result اگر وجود ندارد
    if "_empty_result" not in content:
        empty_func = '''

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
'''
        content += empty_func
        print("✅ تابع _empty_result به analyze_station.py اضافه شد.")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ analyze_station.py در {file_path} به‌روز شد.")


def main():
    print("🚀 شروع اعمال اصلاحات...")
    patch_analyze_station()
    print("✅ اصلاحات کامل شد.")


if __name__ == "__main__":
    main()