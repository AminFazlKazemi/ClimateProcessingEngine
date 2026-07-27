#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_environment.py
============================================
اسکریپت جامع تست محیط climatology_engine
بررسی: مسیرها، فایل‌ها، خواندن داده، حافظه و سرعت
"""

import os
import sys
import time
import glob
import psutil
import numpy as np
import xarray as xr
from pathlib import Path

# =============================================
# تنظیمات اولیه (مشابه main.py)
# =============================================
ZARR_BASE = "I:/climatology_366_rolling"
CONFIG_PATH = r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\config.yaml"
YEARS = list(range(1369, 1400))  # 1369 تا 1399
MONTHS = list(range(1, 13))
VARS = ["tas", "pr", "psl"]  # متغیرهای فرضی، در config.yaml باید بررسی شود

# =============================================
# رنگ‌ها برای خروجی زیبا
# =============================================
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE} {text} {RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️ {text}{RESET}")

def print_info(text):
    print(f"{BLUE}ℹ️ {text}{RESET}")

# =============================================
# ۱. بررسی مسیرها و فایل‌های ورودی
# =============================================
def test_paths():
    print_header("📁 مرحله ۱: بررسی مسیرها و فایل‌ها")
    
    # بررسی پوشه ZARR_BASE
    if os.path.exists(ZARR_BASE):
        print_success(f"پوشه ZARR_BASE وجود دارد: {ZARR_BASE}")
    else:
        print_error(f"پوشه ZARR_BASE پیدا نشد: {ZARR_BASE}")
        return False
    
    # بررسی فایل‌های Zarr موجود
    zarr_files = glob.glob(os.path.join(ZARR_BASE, "*.zarr"))
    if zarr_files:
        print_success(f"تعداد {len(zarr_files)} فایل Zarr پیدا شد:")
        for f in zarr_files[:5]:  # فقط ۵ تا نمایش بده
            size = os.path.getsize(f) / (1024**3)  # تبدیل به گیگابایت
            print(f"   📦 {os.path.basename(f)} ({size:.2f} GB)")
        if len(zarr_files) > 5:
            print(f"   ... و {len(zarr_files) - 5} فایل دیگر")
    else:
        print_warning("هیچ فایل Zarr در مسیر مورد نظر پیدا نشد.")
    
    # بررسی فایل config.yaml
    if os.path.exists(CONFIG_PATH):
        print_success(f"فایل config.yaml وجود دارد: {CONFIG_PATH}")
        try:
            import yaml
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print_info(f"   محتویات config:")
            for key, val in config.items():
                print(f"      {key}: {val}")
        except Exception as e:
            print_warning(f"خطا در خواندن config.yaml: {e}")
    else:
        print_error(f"فایل config.yaml پیدا نشد: {CONFIG_PATH}")
        return False
    
    return True

# =============================================
# ۲. بررسی فایل‌های NetCDF ورودی
# =============================================
def test_netcdf_files():
    print_header("📂 مرحله ۲: بررسی فایل‌های NetCDF")
    
    # مسیر فایل‌های NetCDF را از config بخوانیم (اگر وجود داشت)
    # در غیر این صورت، یک مسیر پیش‌فرض را بررسی می‌کنیم
    data_dir = "K:/gozareshha/dr vazife/140504 - qc temp/climatology/data"
    
    if not os.path.exists(data_dir):
        print_warning(f"پوشه داده‌ها پیدا نشد: {data_dir}")
        print_info("   از داده‌های ساختگی استفاده می‌شود.")
        return True
    
    # پیدا کردن فایل‌های NetCDF
    nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
    if not nc_files:
        print_warning("هیچ فایل NetCDF در پوشه داده‌ها پیدا نشد.")
        print_info("   از داده‌های ساختگی استفاده می‌شود.")
        return True
    
    print_success(f"تعداد {len(nc_files)} فایل NetCDF پیدا شد.")
    
    # بررسی یک فایل نمونه
    sample_file = nc_files[0]
    try:
        ds = xr.open_dataset(sample_file, engine='netcdf4')
        print_success(f"فایل نمونه با موفقیت باز شد: {os.path.basename(sample_file)}")
        print_info(f"   ابعاد: {dict(ds.dims)}")
        print_info(f"   متغیرها: {list(ds.data_vars.keys())}")
        print_info(f"   مختصات: {list(ds.coords.keys())}")
        ds.close()
        return True
    except Exception as e:
        print_error(f"خطا در باز کردن فایل نمونه: {e}")
        return False

# =============================================
# ۳. تست بارگذاری یک بلوک کوچک (اصلاح‌شده)
# =============================================
def test_load_block():
    print_header("⏳ مرحله ۳: تست بارگذاری یک بلوک کوچک")
    
    block_size = 50  # تعداد ایستگاه‌های تست
    block_start = 0
    
    print_info(f"تلاش برای بارگذاری {block_size} ایستگاه...")
    
    # بررسی وجود ماژول‌های مورد نیاز
    try:
        from io_pipeline.read_month_files import read_month_files
        from io_pipeline.assemble_block import assemble_block
        from runtime_tables import build_runtime_tables
        from calendar_tables import build_doy_table_from_config, get_doy_table
        from constants import YEAR_LIST, N_DAYS, N_YEARS, N_VARS, ZARR_BASE
    except ImportError as e:
        print_error(f"خطا در import ماژول‌ها: {e}")
        return False
    
    # ساخت runtime tables با ZARR_BASE
    try:
        tables = build_runtime_tables(ZARR_BASE)
        file_map = tables["file_map"]
        year_list = tables["year_list"]
        print_success("runtime tables با موفقیت ساخته شد.")
        print_info(f"   تعداد فایل‌ها در file_map: {len(file_map)}")
    except Exception as e:
        print_error(f"خطا در ساخت runtime tables: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ساخت doy_table
    try:
        doy_table, _ = build_doy_table_from_config()
        print_success("doy_table با موفقیت ساخته شد.")
        print_info(f"   shape: {doy_table.shape}")
    except Exception as e:
        print_error(f"خطا در ساخت doy_table: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # بارگذاری داده
    start_time = time.time()
    try:
        print_info("   شروع بارگذاری...")
        data_dict = read_month_files(block_start, block_size, file_map, year_list)
        load_time = time.time() - start_time
        print_success(f"بارگذاری با موفقیت انجام شد (زمان: {load_time:.2f} ثانیه)")
        
        # نمایش اطلاعات داده
        print_info(f"   تعداد کلیدهای داده: {len(data_dict)}")
        if data_dict:
            sample_key = next(iter(data_dict))
            sample_data = data_dict[sample_key]
            print_info(f"   نمونه داده: {sample_key} -> shape {sample_data.shape}")
        else:
            print_warning("   داده‌ای بارگذاری نشد!")
            return False
        
        # مونتاژ بلوک
        try:
            block_data = assemble_block(data_dict, doy_table, block_size, year_list)
            print_success(f"مونتاژ بلوک با موفقیت انجام شد (shape: {block_data.shape})")
        except Exception as e:
            print_error(f"خطا در مونتاژ بلوک: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    except Exception as e:
        load_time = time.time() - start_time
        print_error(f"بارگذاری با خطا مواجه شد (زمان: {load_time:.2f} ثانیه)")
        print_error(f"   خطا: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================
# ۴. تست حافظه و عملکرد
# =============================================
def test_memory_and_performance():
    print_header("💾 مرحله ۴: بررسی حافظه و عملکرد")
    
    # RAM کل سیستم
    mem = psutil.virtual_memory()
    print_info(f"   RAM کل: {mem.total / (1024**3):.1f} GB")
    print_info(f"   RAM موجود: {mem.available / (1024**3):.1f} GB")
    print_info(f"   RAM استفاده‌شده: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    
    if mem.percent > 90:
        print_warning(f"⚠️ حافظه بسیار پر است ({mem.percent}%). ممکن است سیستم دچار کندی شود.")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print_info(f"   استفاده از CPU: {cpu_percent}%")
    
    # دیسک
    disk_usage = psutil.disk_usage("I:")
    print_info(f"   فضای دیسک I: {disk_usage.free / (1024**3):.1f} GB موجود (از {disk_usage.total / (1024**3):.1f} GB)")
    if disk_usage.free / (1024**3) < 10:
        print_warning("   ⚠️ فضای دیسک کمتر از ۱۰ گیگابایت است. ممکن است برای Zarr کافی نباشد.")
    
    return True

# =============================================
# ۵. تست سرعت I/O
# =============================================
def test_io_speed():
    print_header("⚡ مرحله ۵: تست سرعت I/O (خواندن/نوشتن)")
    
    # تست نوشتن یک فایل موقت
    test_file = "I:/io_test_temp.tmp"
    try:
        import tempfile
        import shutil
        
        # تست نوشتن
        start_time = time.time()
        with open(test_file, 'wb') as f:
            f.write(b'0' * 100 * 1024 * 1024)  # 100 MB
        write_time = time.time() - start_time
        print_success(f"نوشتن ۱۰۰ مگابایت: {write_time:.2f} ثانیه ({100/write_time:.1f} MB/s)")
        
        # تست خواندن
        start_time = time.time()
        with open(test_file, 'rb') as f:
            data = f.read()
        read_time = time.time() - start_time
        print_success(f"خواندن ۱۰۰ مگابایت: {read_time:.2f} ثانیه ({100/read_time:.1f} MB/s)")
        
        # پاک کردن فایل تست
        os.remove(test_file)
        
        # مقایسه با حداقل سرعت قابل قبول
        if write_time > 10 or read_time > 10:
            print_warning("⚠️ سرعت I/O پایین است. ممکن است پردازش کند باشد.")
        else:
            print_success("سرعت I/O قابل قبول است.")
    except Exception as e:
        print_error(f"خطا در تست I/O: {e}")
        return False
    
    return True

# =============================================
# ۶. تست Dask (در صورت وجود)
# =============================================
def test_dask():
    print_header("🧵 مرحله ۶: تست Dask")
    
    try:
        import dask
        from dask.distributed import Client
        
        # نسخه Dask
        print_info(f"   نسخه Dask: {dask.__version__}")
        
        # راه‌اندازی Client
        try:
            client = Client(processes=False, threads_per_worker=2, timeout="10s")
            print_success(f"Dask Client راه‌اندازی شد: {client}")
            client.close()
        except Exception as e:
            print_warning(f"خطا در راه‌اندازی Dask Client: {e}")
            return False
    except ImportError:
        print_warning("Dask نصب نیست. این تأثیری روی پردازش عادی ندارد.")
        return True
    
    return True

# =============================================
# ۷. تست پیشرفت و لاگ (مشکل نمایش)
# =============================================
def test_progress_logging():
    print_header("📊 مرحله ۷: تست نمایش پیشرفت")
    
    # بررسی تنظیمات logger
    try:
        from monitoring.logger import logger
        print_success("logger با موفقیت بارگذاری شد.")
    except ImportError as e:
        print_warning(f"خطا در بارگذاری logger: {e}")
        return True
    
    # تست چاپ یک پیام نمونه
    logger.info("🧪 این یک پیام تست از logger است.")
    print_success("پیام تست با موفقیت ارسال شد.")
    
    # بررسی سطح لاگ
    import logging
    current_level = logger.level
    print_info(f"   سطح لاگ فعلی: {logging.getLevelName(current_level)}")
    
    if current_level > logging.INFO:
        print_warning(f"سطح لاگ بالاتر از INFO است ({logging.getLevelName(current_level)}). ممکن است برخی پیام‌ها نمایش داده نشوند.")
    
    return True

# =============================================
# ۸. خلاصه و پیشنهادات
# =============================================
def print_summary(results):
    print_header("📋 خلاصه نهایی")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print_info(f"تعداد تست‌های موفق: {success_count} از {total_count}")
    
    if success_count == total_count:
        print_success("🎉 همه تست‌ها با موفقیت انجام شد! محیط شما آماده است.")
        print_info("   اگر پردازش هنوز کند است، می‌توانید block_size را در main.py کاهش دهید.")
    else:
        print_warning("⚠️ بعضی از تست‌ها با مشکل مواجه شدند.")
        print_info("   موارد زیر نیاز به بررسی دارند:")
        for test_name, result in results.items():
            if not result:
                print(f"   - {test_name}")
    
    # پیشنهادات
    print_header("💡 پیشنهادات نهایی")
    print("1. اگر پردازش در مرحله 'Loading' گیر کرد، block_size را به 50 کاهش دهید.")
    print("2. برای سرعت بیشتر، 'VALIDATE_AFTER_LOAD' را در config.yaml به False تغییر دهید.")
    print("3. مطمئن شوید که فضای دیسک کافی (حداقل 20 گیگابایت) وجود دارد.")
    print("4. اگر مشکل حافظه دارید، تعداد ایستگاه‌های پردازش هم‌زمان را کاهش دهید.")

# =============================================
# اجرای اصلی
# =============================================
def main():
    print_header("🧪 شروع تست کامل محیط climatology_engine")
    
    results = {}
    
    # اجرای تست‌ها
    results["مسیرها و فایل‌ها"] = test_paths()
    results["فایل‌های NetCDF"] = test_netcdf_files()
    results["بارگذاری بلوک"] = test_load_block()
    results["حافظه و عملکرد"] = test_memory_and_performance()
    results["سرعت I/O"] = test_io_speed()
    results["Dask"] = test_dask()
    results["نمایش پیشرفت"] = test_progress_logging()
    
    # خلاصه نهایی
    print_summary(results)
    
    print_header("🏁 پایان تست")

if __name__ == "__main__":
    main()