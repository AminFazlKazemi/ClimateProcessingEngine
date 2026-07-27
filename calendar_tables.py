#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calendar_tables.py
============================================
ساخت جدول روزهای سال (doy_table) از فایل calendar.txt یا fallback
"""

import os
import numpy as np
from constants import YEAR_START, YEAR_END, N_YEARS, N_DAYS, CALENDAR_FILE

# =============================================
# سال‌های کبیسه شمسی (برای fallback)
# =============================================
PERSIAN_LEAP_YEARS = [
    1375, 1379, 1383, 1387, 1391, 1395, 1399,
    1403, 1407, 1411, 1415, 1419, 1423, 1427,
]

def is_persian_leap(year):
    return year in PERSIAN_LEAP_YEARS

def persian_days_before_month(year, month):
    """تعداد روزهای قبل از ماه مورد نظر در سال شمسی"""
    days_in_months = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]
    if is_persian_leap(year):
        days_in_months[11] = 30
    return sum(days_in_months[:month - 1]) + 1  # +1 چون روز از 1 شروع می‌شود

def build_doy_table_from_file(filepath):
    """ساخت doy_table از فایل calendar.txt"""
    calendar_lookup = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            shamsi_str = parts[0]
            gregorian_str = parts[1]  # استفاده نمی‌شود
            shamsi_doy = int(parts[2])
            year = int(shamsi_str[:4])
            month = int(shamsi_str[4:6])
            day = int(shamsi_str[6:8])
            calendar_lookup[(year, month, day)] = shamsi_doy

    doy_table = np.full((N_YEARS, 13, 32), -1, dtype=np.int16)
    year_index = {year: idx for idx, year in enumerate(range(YEAR_START, YEAR_END + 1))}
    for (year, month, day), doy in calendar_lookup.items():
        if year in year_index:
            year_idx = year_index[year]
            doy_table[year_idx, month, day] = doy
    return doy_table

def build_doy_table_fallback():
    """ساخت doy_table به‌صورت fallback (بدون فایل)"""
    doy_table = np.full((N_YEARS, 13, 32), -1, dtype=np.int16)
    year_list = list(range(YEAR_START, YEAR_END + 1))
    for year_idx, year in enumerate(year_list):
        for month in range(1, 13):
            if month <= 6:
                days = 31
            elif month < 12:
                days = 30
            else:
                days = 29 if is_persian_leap(year) else 30
            for day in range(1, days + 1):
                doy = persian_days_before_month(year, month) + day - 1
                doy_table[year_idx, month, day] = doy
    return doy_table

def build_doy_table_from_config():
    """ساخت doy_table از فایل config (استفاده از CALENDAR_FILE)"""
    if os.path.exists(CALENDAR_FILE):
        try:
            doy_table = build_doy_table_from_file(CALENDAR_FILE)
            print(f"   ✅ doy_table از {CALENDAR_FILE} ساخته شد.")
            return doy_table, "calendar.txt"
        except Exception as e:
            print(f"   ⚠️ خطا در خواندن {CALENDAR_FILE}: {e}")
            print("   استفاده از fallback...")
            doy_table = build_doy_table_fallback()
            return doy_table, "fallback"
    else:
        print(f"   ⚠️ {CALENDAR_FILE} یافت نشد. استفاده از fallback.")
        doy_table = build_doy_table_fallback()
        return doy_table, "fallback"

def get_doy_table():
    """دریافت doy_table با کش (برای استفاده در سایر ماژول‌ها)"""
    if not hasattr(get_doy_table, "_cache"):
        doy_table, source = build_doy_table_from_config()
        get_doy_table._cache = doy_table
        get_doy_table._source = source
    return get_doy_table._cache

# =============================================
# اجرای آزمایشی
# =============================================
if __name__ == "__main__":
    doy_table, source = build_doy_table_from_config()
    print(f"✅ doy_table shape: {doy_table.shape}")
    print(f"   منبع: {source}")