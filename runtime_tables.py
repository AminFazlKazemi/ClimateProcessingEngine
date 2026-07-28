#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_tables.py
============================================
ساخت جداول زمان اجرا:
- window_table: پنجره‌های هر روز سال
- file_map: نقشه فایل‌های سالانه/ماهانه
- variable_table: اطلاعات متغیرها
"""

import os
import glob
import numpy as np
from constants import (
    YEAR_LIST, N_YEARS, N_DAYS, WINDOW_DAYS, VARS, ZARR_BASE
)

# =============================================
# ۱. ساخت جدول پنجره‌ها (Window Table)
# =============================================
def build_window_table():
    N_DAYS_LOCAL = 366
    WINDOW_DAYS_LOCAL = 2
    window_table = []
    for day in range(N_DAYS_LOCAL):
        window = [(day + offset) % N_DAYS_LOCAL for offset in range(-WINDOW_DAYS_LOCAL, WINDOW_DAYS_LOCAL + 1)]
        window_table.append(window)
    return window_table
# =============================================
# ۲. ساخت نقشه فایل‌ها (File Map)
# =============================================
def build_file_map(zarr_base):
    """ساخت نقشه فایل‌های Zarr بر اساس سال و ماه"""
    file_map = {}
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    for f in zarr_files:
        basename = os.path.basename(f)
        parts = basename.replace(".zarr", "").split("_")
        if len(parts) >= 2:
            try:
                year = int(parts[0])
                month = int(parts[1])
                if year in YEAR_LIST and 1 <= month <= 12:
                    file_map[(year, month)] = f
            except ValueError:
                continue
    return file_map

# =============================================
# ۳. ساخت جدول متغیرها (Variable Table)
# =============================================
def build_variable_table():
    """ساخت جدول متغیرها با ایندکس‌ها"""
    var_index = {name: idx for idx, name in enumerate(VARS)}
    return {
        "var_index": var_index,
        "var_names": VARS,
        "n_vars": len(VARS)
    }

# =============================================
# ۴. ساخت همه جداول (تابع اصلی)
# =============================================

def build_runtime_tables(zarr_base):
    """ساخت جداول زمان اجرا از فایل‌های موجود Zarr"""
    import glob
    import os
    import re
    from constants import WINDOW_DAYS, N_DAYS, YEAR_START, YEAR_END

    # 1. پیدا کردن تمام فایل‌های Zarr
    pattern = os.path.join(zarr_base, "*.zarr")
    zarr_files = glob.glob(pattern)

    if not zarr_files:
        raise FileNotFoundError(f"No Zarr files found in {zarr_base}")

    # 2. ساخت file_map با استخراج سال و ماه از نام فایل
    file_map = {}
    years_set = set()
    for f in zarr_files:
        basename = os.path.basename(f)
        # نام فایل: 1369_01_Farvardin.zarr یا 1370_02.zarr
        # استخراج سال و ماه
        parts = basename.replace(".zarr", "").split("_")
        if len(parts) >= 2:
            try:
                year = int(parts[0])
                month = int(parts[1])
                file_map[(year, month)] = f
                years_set.add(year)
            except ValueError:
                continue

    if not file_map:
        raise FileNotFoundError(f"No valid Zarr files with year/month found in {zarr_base}")

    # 3. ساخت year_list از سال‌های موجود
    year_list = sorted(years_set)

    # 4. ساخت window_table
    window_table = []
    for day in range(N_DAYS):  # 0-based day indices
        window = [(day + offset) % N_DAYS for offset in range(-WINDOW_DAYS, WINDOW_DAYS + 1)]
        window_table.append(window)
    for day in range(1, N_DAYS + 1):
        window = [(day + offset - 1) % N_DAYS + 1 for offset in range(-WINDOW_DAYS, WINDOW_DAYS + 1)]
        window_table.append(window)

    return {
        "file_map": file_map,
        "window_table": window_table,
        "year_list": year_list
    }

def print_runtime_info(tables):
    print(f"   window_table: {len(tables['window_table'])} روز")
    print(f"   file_map: {len(tables['file_map'])} فایل")
    print(f"   variables: {tables['variable_table']['n_vars']} متغیر")
    print(f"   سال‌ها: {tables['year_list'][0]} تا {tables['year_list'][-1]}")

# =============================================
# اجرای آزمایشی
# =============================================
if __name__ == "__main__":
    tables = build_runtime_tables()
    print("✅ Runtime Tables:")
    print_runtime_info(tables)