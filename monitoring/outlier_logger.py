#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitoring/outlier_logger.py
ثبت داده‌های پرت در حین پردازش با قابلیت flush روی دیسک
"""

import os
import csv
from datetime import datetime
from collections import defaultdict

# دیکشنری سراسری برای ذخیره پرت‌ها
_outlier_store = defaultdict(list)

def log_outlier(station_idx, day_idx, value, var_name="tmean"):
    """
    ثبت یک داده‌ی پرت در حافظه
    """
    _outlier_store[station_idx].append({
        'day': day_idx,
        'value': float(value),
        'var': var_name,
        'timestamp': datetime.now().isoformat()
    })

def get_outlier_count(station_idx):
    """تعداد پرت‌های ثبت‌شده برای یک ایستگاه"""
    return len(_outlier_store.get(station_idx, []))

def get_outlier_summary():
    """گرفتن خلاصه‌ای از همه پرت‌ها"""
    return dict(_outlier_store)

def flush_outliers_to_csv(output_dir="outlier_reports", metadata_path=None):
    """
    نوشتن تمام پرت‌های موجود در حافظه به فایل CSV (حالت append)
    و سپس پاک کردن حافظه.
    
    این تابع را می‌توان بعد از هر بلوک فراخوانی کرد تا داده‌ها
    به‌طور مداوم روی دیسک ذخیره شوند.
    """
    global _outlier_store
    if not _outlier_store:
        return

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "outliers.csv")

    # بارگذاری متادیتا (در صورت وجود)
    station_meta = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            import pandas as pd
            df = pd.read_csv(metadata_path, sep='\t')
            for _, row in df.iterrows():
                station_meta[row['stationid']] = {
                    'new_part': row.get('new_part', ''),
                    'shahrestan': row.get('shahrestan', ''),
                    'subbasin_i': row.get('subbasin_i', ''),
                    'Longitude': row.get('Longitude', ''),
                    'Latitude': row.get('Latitude', ''),
                    'new_elevation': row.get('new_elevation', '')
                }
        except Exception as e:
            print(f"⚠️ خطا در خواندن metadata: {e}")

    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # نوشتن هدر فقط در صورتی که فایل تازه ایجاد شده باشد
        if not file_exists:
            header = ['station_idx', 'day', 'value', 'var', 'timestamp']
            if station_meta:
                header += ['new_part', 'shahrestan', 'subbasin_i', 'Longitude', 'Latitude', 'new_elevation']
            writer.writerow(header)

        total_written = 0
        # نوشتن همه رکوردها
        for station_idx, outliers in list(_outlier_store.items()):
            meta = station_meta.get(station_idx, {})
            for out in outliers:
                row = [
                    station_idx,
                    out['day'],
                    out['value'],
                    out['var'],
                    out['timestamp']
                ]
                if station_meta:
                    row += [
                        meta.get('new_part', ''),
                        meta.get('shahrestan', ''),
                        meta.get('subbasin_i', ''),
                        meta.get('Longitude', ''),
                        meta.get('Latitude', ''),
                        meta.get('new_elevation', '')
                    ]
                writer.writerow(row)
                total_written += 1

    # پاک کردن حافظه بعد از نوشتن روی دیسک
    count_before = sum(len(v) for v in _outlier_store.values())
    _outlier_store.clear()
    print(f"💾 Flushed {count_before} outlier records to {filepath}")

def save_outlier_report(output_dir="outlier_reports", metadata_path=None):
    """ذخیره نهایی (همان flush) – برای سازگاری با کدهای قدیمی"""
    flush_outliers_to_csv(output_dir, metadata_path)

def clear_outlier_log():
    """پاک کردن حافظه (بدون نوشتن روی دیسک)"""
    global _outlier_store
    _outlier_store = defaultdict(list)