#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_status.py – بررسی وضعیت فعلی پردازش
- وضعیت فایل Zarr (تعداد نقاط، زمان آخرین تغییر)
- وضعیت checkpoint (بلوک و ایستگاه آخر)
- وضعیت کش دیسک (حجم، تعداد فایل‌ها)
- لاگ‌های اخیر (آخرین پیام‌ها)
"""

import os
import sys
import time
import glob
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# مسیرها (همانند config.yaml)
# ============================================================================
BASE_DIR = Path(__file__).parent
ZARR_PATH = Path(r"I:/climatology_366_rolling/climatology_stationwise_final.zarr")
CHECKPOINT_FILE = BASE_DIR / "nature_output" / "checkpoint.csv"
LOG_FILE = BASE_DIR / "logs" / "climatology.log"
CACHE_DIR = BASE_DIR / "cache" / "zarr_cache"

print("=" * 80)
print("🔍 تشخیص وضعیت پردازش")
print("=" * 80)

# ============================================================================
# ۱. وضعیت فایل Zarr
# ============================================================================
print("\n📂 ۱. وضعیت فایل Zarr:")
if ZARR_PATH.exists():
    stat = ZARR_PATH.stat()
    size_gb = stat.st_size / (1024**3)
    last_modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"   ✅ فایل وجود دارد")
    print(f"   حجم: {size_gb:.2f} GB")
    print(f"   آخرین تغییر: {last_modified}")
    
    # بررسی تعداد نقاط با xarray (اگر خوانده شود)
    try:
        import xarray as xr
        ds = xr.open_zarr(ZARR_PATH, consolidated=False)
        n_points = ds.dims.get("point", 0)
        n_days = ds.dims.get("day", 0)
        print(f"   تعداد نقاط در Zarr: {n_points:,}")
        print(f"   تعداد روزها: {n_days}")
        # بررسی اینکه آیا داده‌ای برای نقاط جدید نوشته شده
        if n_points > 0:
            # بررسی یک متغیر نمونه برای نقطه آخر (اگر وجود داشته باشد)
            last_point_data = ds["tmean_mean"].isel(point=-1).values
            valid_count = (~np.isnan(last_point_data)).sum()
            print(f"   نقطه‌ی آخر ({n_points-1}) دارای {valid_count} روز معتبر از {n_days}")
            if valid_count == 0:
                print("      ⚠️ نقطه‌ی آخر کاملاً NaN است (هنوز نوشته نشده)")
            elif valid_count < n_days:
                print("      ⚠️ نقطه‌ی آخر فقط تا حدی پر شده (نوشتن ناقص)")
            else:
                print("      ✅ نقطه‌ی آخر کامل است")
        ds.close()
    except Exception as e:
        print(f"   ⚠️ خطا در خواندن Zarr: {e}")
else:
    print("   ❌ فایل Zarr وجود ندارد")

# ============================================================================
# ۲. وضعیت Checkpoint
# ============================================================================
print("\n📌 ۲. وضعیت Checkpoint:")
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, 'r') as f:
        lines = f.read().strip().split('\n')
    data = {}
    for line in lines:
        if '=' in line:
            k, v = line.split('=', 1)
            data[k] = v
    block = data.get("block", "نامشخص")
    station = data.get("station", "نامشخص")
    timestamp = data.get("timestamp", "نامشخص")
    print(f"   ✅ فایل checkpoint وجود دارد")
    print(f"   آخرین بلوک: {block}")
    print(f"   آخرین ایستگاه: {station}")
    if timestamp != "نامشخص":
        try:
            dt = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"   زمان ذخیره: {dt}")
        except:
            print(f"   زمان ذخیره (خام): {timestamp}")
else:
    print("   ℹ️ فایل checkpoint وجود ندارد")

# ============================================================================
# ۳. وضعیت کش دیسک
# ============================================================================
print("\n💾 ۳. وضعیت کش دیسک:")
if CACHE_DIR.exists():
    cache_files = list(CACHE_DIR.glob("*.npy"))
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files) / (1024**2)
        print(f"   تعداد فایل‌های کش: {len(cache_files)}")
        print(f"   حجم کل کش: {total_size:.2f} MB")
        # آخرین فایل کش
        latest = max(cache_files, key=lambda f: f.stat().st_mtime)
        latest_time = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"   آخرین کش: {latest.name} ({latest_time})")
        # بررسی اینکه آیا کش‌های جدید ساخته می‌شوند یا نه
        if len(cache_files) > 0:
            print("   ✅ کش در حال استفاده است")
    else:
        print("   ℹ️ کش دیسک خالی است")
else:
    print("   ℹ️ پوشه‌ی کش وجود ندارد")

# ============================================================================
# ۴. لاگ‌های اخیر
# ============================================================================
print("\n📄 ۴. بررسی لاگ‌های اخیر (۱۰ خط آخر):")
if LOG_FILE.exists():
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if lines:
        print("   آخرین ۱۰ خط لاگ:")
        for line in lines[-10:]:
            print(f"      {line.strip()}")
    else:
        print("   ℹ️ لاگ خالی است")
else:
    print("   ❌ فایل لاگ وجود ندارد")

# ============================================================================
# ۵. بررسی فرآیندهای در حال اجرا (اختیاری)
# ============================================================================
print("\n🔄 ۵. بررسی فرآیندهای پایتون در حال اجرا:")
try:
    import psutil
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'] and 'python' in proc.info['name'].lower():
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'main.py' in cmdline or 'climatology' in cmdline:
                python_procs.append({
                    'pid': proc.info['pid'],
                    'cmdline': cmdline[:100],
                    'cpu_percent': proc.cpu_percent(interval=0.1),
                    'memory_mb': proc.memory_info().rss / (1024**2)
                })
    if python_procs:
        print(f"   تعداد فرآیندهای مرتبط: {len(python_procs)}")
        for p in python_procs:
            print(f"   PID {p['pid']}: CPU {p['cpu_percent']:.1f}%, RAM {p['memory_mb']:.0f} MB")
            print(f"      {p['cmdline']}")
    else:
        print("   ℹ️ هیچ فرآیند پایتون مرتبطی یافت نشد")
except ImportError:
    print("   ⚠️ psutil نصب نیست (برای اطلاعات دقیق‌تر نصب کنید: pip install psutil)")
except Exception as e:
    print(f"   ⚠️ خطا در بررسی فرآیندها: {e}")

# ============================================================================
# ۶. جمع‌بندی
# ============================================================================
print("\n" + "=" * 80)
print("📊 جمع‌بندی:")
if ZARR_PATH.exists() and CHECKPOINT_FILE.exists():
    try:
        block_val = int(data.get("block", -1))
        station_val = int(data.get("station", -1))
        # اگر checkpoint نشان می‌دهد که به نقاط جدید رسیده
        if station_val > 40000:
            print("✅ برنامه به نقاط جدید (بیش از ۴۰,۰۰۰) رسیده است.")
            print(f"   آخرین ایستگاه: {station_val:,}")
        else:
            print("⚠️ برنامه هنوز به نقاط جدید نرسیده است (ایستگاه ≤ ۴۰,۰۰۰).")
            print(f"   آخرین ایستگاه: {station_val:,}")
        # بررسی زمان آخرین بروزرسانی Zarr
        if ZARR_PATH.exists():
            last_mod = datetime.fromtimestamp(ZARR_PATH.stat().st_mtime)
            now = datetime.now()
            diff = (now - last_mod).total_seconds() / 60
            if diff < 5:
                print(f"✅ Zarr به‌روز است (آخرین تغییر {diff:.1f} دقیقه پیش)")
            else:
                print(f"⚠️ Zarr مدت‌هاست به‌روز نشده ({diff:.1f} دقیقه). ممکن است نوشتن متوقف شده باشد.")
    except:
        pass
else:
    print("⚠️ اطلاعات کافی برای جمع‌بندی وجود ندارد")

print("=" * 80)