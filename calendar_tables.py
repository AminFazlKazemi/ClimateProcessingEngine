# -*- coding: utf-8 -*-
"""
calendar_tables.py - ساخت جداول تقویم جلالی
(نسخه اصلاح‌شده با پشتیبانی از سرستون و خطای `np.str_('miladi')`)
"""

import os
import numpy as np
from constants import CALENDAR_FILE, YEAR_START, YEAR_END, N_YEARS, N_DAYS

def build_doy_table_from_config():
    """ساخت doy_table از روی فایل calendar.txt (برای سال‌های پیش‌فرض)"""
    return build_doy_table_for_years(list(range(YEAR_START, YEAR_END + 1)))

def build_doy_table_for_years(year_list):
    """
    ساخت doy_table برای سال‌های مشخص با استفاده از calendar.txt یا فرمول

    Parameters:
        year_list: list of Persian years (e.g., [1369, 1370, ...])

    Returns:
        doy_table: ndarray (len(year_list), 13, 32) – day-of-year mapping
        day_names: None (for compatibility)
    """
    year_list = sorted(year_list)
    n_years = len(year_list)
    year_to_idx = {year: idx for idx, year in enumerate(year_list)}

    # 1. Try to use calendar.txt
    doy_table = None
    if os.path.exists(CALENDAR_FILE):
        try:
            # ============================================================
            # اصلاح: خواندن با skiprows=1 برای رد کردن سرستون
            # ============================================================
            data = np.loadtxt(CALENDAR_FILE, dtype=str, delimiter=None, encoding='utf-8', skiprows=1)
            
            # Parse ShamsiDate: first 4 digits = year, next 2 = month, last 2 = day
            dates = data[:, 0]
            doys = data[:, 3].astype(int)  # julian_date

            # Create doy_table
            doy_table = np.zeros((n_years, 13, 32), dtype=np.int32)
            filled = 0
            for idx, year in enumerate(year_list):
                # Find rows for this year
                year_mask = np.array([int(d[:4]) == year for d in dates])
                rows = data[year_mask]
                if len(rows) == 0:
                    continue
                for row in rows:
                    date_str = row[0]
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    doy = int(row[3])  # ستون julian_date
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        doy_table[idx, month, day] = doy
                        filled += 1
            if filled > 0:
                print(f"   ✅ doy_table از {CALENDAR_FILE} با {n_years} سال ساخته شد.")
                return doy_table, None
        except Exception as e:
            print(f"   ⚠️ خطا در خواندن {CALENDAR_FILE}: {e}")

    # 2. Fallback: build from formula (simple Persian calendar)
    if doy_table is None:
        doy_table = np.zeros((n_years, 13, 32), dtype=np.int32)
        for idx, year in enumerate(year_list):
            doy = 1
            for month in range(1, 13):
                days_in_month = get_persian_month_days(month, year)
                for day in range(1, days_in_month + 1):
                    doy_table[idx, month, day] = doy
                    doy += 1
        print(f"   ⚠️ از فرمول تقویم جلالی برای {n_years} سال استفاده شد.")
        return doy_table, None

def is_persian_leap_year(year):
    """بررسی کبیسه بودن سال جلالی"""
    return (year + 38) % 4 == 0

def get_persian_month_days(month, year):
    """تعداد روزهای ماه جلالی"""
    if month <= 6:
        return 31
    elif month <= 11:
        return 30
    else:  # month == 12
        return 30 if is_persian_leap_year(year) else 29