#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calendar_tables.py
================================================================================
ساخت جداول تقویمی مستقل از فایل.
تنها وابستگی به calendar.txt دارد و در غیر این صورت از لیست کبیسه‌ها استفاده می‌کند.
================================================================================
ورژن: 2.0 - نهایی
"""

import os  # <--- این خط را اضافه کنید
import numpy as np
from constants import YEAR_START, YEAR_END, N_YEARS, CALENDAR_FILE

# ============================================================================
# لیست سال‌های کبیسه شمسی (Fallback)
# ============================================================================
PERSIAN_LEAP_YEARS = {
    1329, 1333, 1337, 1342, 1346, 1350, 1354, 1358, 1362, 1366,
    1370, 1375, 1379, 1383, 1387, 1391, 1395, 1399, 1403, 1408,
    1412, 1416, 1420, 1424, 1428, 1432, 1436, 1440, 1444, 1448,
    1452, 1456, 1460, 1464,
}

# ============================================================================
# توابع پایه تقویم
# ============================================================================
def is_persian_leap_year(year):
    return year in PERSIAN_LEAP_YEARS

def get_persian_month_days(month, year):
    if month <= 6:
        return 31
    elif month <= 11:
        return 30
    else:
        return 30 if is_persian_leap_year(year) else 29

def get_shamsi_doy(year, month, day):
    days_in_months = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]
    if is_persian_leap_year(year):
        days_in_months[11] = 30
    return sum(days_in_months[:month - 1]) + day

# ============================================================================
# ساخت جدول doy
# ============================================================================
def build_doy_table():
    doy_table = np.full((N_YEARS, 13, 32), -1, dtype=np.int16)
    for year_idx, year in enumerate(range(YEAR_START, YEAR_END + 1)):
        for month in range(1, 13):
            days = get_persian_month_days(month, year)
            for day in range(1, days + 1):
                doy = get_shamsi_doy(year, month, day)
                doy_table[year_idx, month, day] = doy - 1  # 0-based
    return doy_table

def build_year_index():
    return {year: idx for idx, year in enumerate(range(YEAR_START, YEAR_END + 1))}

def build_year_list():
    return list(range(YEAR_START, YEAR_END + 1))

# ============================================================================
# خواندن calendar.txt
# ============================================================================
def read_calendar_file(filepath):
    if not os.path.exists(filepath):
        return None
    calendar_lookup = {}
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            shamsi_str = parts[0]
            shamsi_doy = int(parts[2])
            year = int(shamsi_str[:4])
            month = int(shamsi_str[4:6])
            day = int(shamsi_str[6:8])
            calendar_lookup[(year, month, day)] = shamsi_doy - 1
    return calendar_lookup

def build_doy_table_from_calendar(calendar_lookup):
    doy_table = np.full((N_YEARS, 13, 32), -1, dtype=np.int16)
    year_index = build_year_index()
    for (year, month, day), doy in calendar_lookup.items():
        if year in year_index:
            year_idx = year_index[year]
            if 1 <= month <= 12 and 1 <= day <= 31:
                doy_table[year_idx, month, day] = doy
    return doy_table

# ============================================================================
# تابع اصلی
# ============================================================================
def build_calendar_tables():
    calendar_lookup = read_calendar_file(CALENDAR_FILE)
    if calendar_lookup is not None:
        doy_table = build_doy_table_from_calendar(calendar_lookup)
        print(f"   ✅ doy_table از calendar.txt ساخته شد.")
    else:
        doy_table = build_doy_table()
        print(f"   ⚠️ calendar.txt یافت نشد. استفاده از Fallback.")
    year_index = build_year_index()
    year_list = build_year_list()
    return {
        "doy_table": doy_table,
        "year_index": year_index,
        "year_list": year_list,
        "source": "calendar.txt" if calendar_lookup is not None else "fallback",
    }

if __name__ == "__main__":
    tables = build_calendar_tables()
    doy_table = tables["doy_table"]
    print(f"✅ doy_table shape: {doy_table.shape}")
    print(f"   سال‌ها: {tables['year_list'][0]} تا {tables['year_list'][-1]}")
    print(f"   منبع: {tables['source']}")