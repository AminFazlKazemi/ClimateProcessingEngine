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
    """ساخت جدول روزهای پنجره برای هر روز سال"""
    window_table = []
    for doy in range(N_DAYS):
        indices = []
        for d in range(doy - WINDOW_DAYS, doy + WINDOW_DAYS + 1):
            # تنظیم دامنه روز (0 تا N_DAYS-1)
            d_adj = d % N_DAYS
            if d_adj < 0:
                d_adj += N_DAYS
            indices.append(d_adj)
        window_table.append(indices)
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
def build_runtime_tables(zarr_base=None):
    """
    ساخت تمام جداول زمان اجرا
    Args:
        zarr_base: مسیر پایه فایل‌های Zarr (در صورت None، از constants.ZARR_BASE استفاده می‌کند)
    Returns:
        dict: شامل window_table, file_map, variable_table
    """
    if zarr_base is None:
        zarr_base = ZARR_BASE
    
    tables = {
        "window_table": build_window_table(),
        "file_map": build_file_map(zarr_base),
        "variable_table": build_variable_table(),
        "year_list": YEAR_LIST,
        "n_years": N_YEARS,
        "n_days": N_DAYS,
        "vars": VARS,
    }
    return tables

# =============================================
# ۵. نمایش اطلاعات (برای دیباگ)
# =============================================
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