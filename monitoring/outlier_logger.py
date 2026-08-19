#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitoring/outlier_logger.py
ثبت داده‌های پرت با پشتیبانی از سال و اسکیل خودکار دما
نسخه اصلاح‌شده برای ثبت year و تقسیم بر ۱۰
"""

import os
import csv
from datetime import datetime
from collections import defaultdict

_outlier_store = defaultdict(list)

def log_outlier(station_idx, year, day_idx, value, var_name="tmean"):
    """
    ثبت یک داده‌ی پرت در حافظه.
    
    پارامترها:
        station_idx: ایندکس ایستگاه (0-based) - در زمان flush به stationid تبدیل می‌شود
        year: سال وقوع
        day_idx: روز سال (1 تا 366)
        value: مقدار پرت
        var_name: نام متغیر ('tmin', 'tmean', 'tmax')
    """
    # اسکیل خودکار برای tmean: اگر مقدار > 50 بود، بر ۱۰ تقسیم کن
    final_value = float(value)
    if var_name == 'tmean' and final_value > 50:
        final_value = final_value / 10.0

    _outlier_store[station_idx].append({
        'year': year,
        'day': day_idx,
        'value': final_value,
        'var': var_name,
        'timestamp': datetime.now().isoformat()
    })

def get_outlier_count(station_idx):
    return len(_outlier_store.get(station_idx, []))

def get_outlier_summary():
    return dict(_outlier_store)

def flush_outliers_to_csv(output_dir="outlier_reports", metadata_path=None):
    global _outlier_store
    if not _outlier_store:
        return

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "outliers.csv")

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

        if not file_exists:
            header = ['station_idx', 'year', 'day', 'value', 'var', 'timestamp']
            if station_meta:
                header += ['new_part', 'shahrestan', 'subbasin_i', 'Longitude', 'Latitude', 'new_elevation']
            writer.writerow(header)

        total_written = 0
        for station_idx, outliers in list(_outlier_store.items()):
            meta = station_meta.get(station_idx, {})
            for out in outliers:
                row = [
                    station_idx,
                    out['year'],
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

    count_before = sum(len(v) for v in _outlier_store.values())
    _outlier_store.clear()
    print(f"💾 Flushed {count_before} outlier records to {filepath}")

def save_outlier_report(output_dir="outlier_reports", metadata_path=None):
    flush_outliers_to_csv(output_dir, metadata_path)

def clear_outlier_log():
    global _outlier_store
    _outlier_store = defaultdict(list)