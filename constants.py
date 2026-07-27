#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
constants.py
============================================
تعاریف ثابت‌های اصلی برنامه
تمام مقادیر از config.yaml خوانده می‌شوند.
"""

import os
import yaml
import numpy as np

# =============================================
# بارگذاری فایل config.yaml
# =============================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config(CONFIG_PATH)

# =============================================
# ۱. تنظیمات سال و روز
# =============================================
YEAR_START = CONFIG["years"]["start"]
YEAR_END = CONFIG["years"]["end"]
YEAR_LIST = list(range(YEAR_START, YEAR_END + 1))   # [1369, 1370, ..., 1399]
N_YEARS = len(YEAR_LIST)                            # 31
N_DAYS = CONFIG["days"]                             # 366

# =============================================
# ۲. تنظیمات متغیرها
# =============================================
VARS = CONFIG["variables"]                          # ['tmin', 'tmean', 'tmax']
N_VARS = len(VARS)                                  # 3
VAR_INDEX = {name: idx for idx, name in enumerate(VARS)}
VAR_INDEX_FOR_FIT = CONFIG["fit_variable_index"]    # معمولاً 1 (tmean)

# =============================================
# ۳. تنظیمات پنجره (Window)
# =============================================
WINDOW_DAYS = CONFIG["window_days"]                 # 2 (یعنی ±2 روز)

# =============================================
# ۴. تنظیمات مسیرها (Paths)
# =============================================
PATHS = CONFIG["paths"]
OUTPUT_DIR = PATHS["output_dir"]
OUTPUT_ZARR = os.path.join(OUTPUT_DIR, PATHS["output_zarr_name"])
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, PATHS["checkpoint_file"])
ZARR_BASE = PATHS["zarr_base"]
CALENDAR_FILE = PATHS["calendar_file"]

# =============================================
# ۵. تنظیمات پردازش
# =============================================
BLOCK_SIZE = CONFIG["block_size"]                   # 5000 (پیش‌فرض جدید)
USE_PARALLEL = CONFIG["use_parallel"]               # True
CORES = CONFIG["cores"]                             # 6

# =============================================
# ۶. تنظیمات اعتبارسنجی (Validation)
# =============================================
VALIDATION = CONFIG["validation"]
VALIDATE_AFTER_LOAD = VALIDATION["validate_after_load"]
VALIDATE_BEFORE_WRITE = VALIDATION["validate_before_write"]
VALIDATE_EVERY_N_BLOCKS = VALIDATION["validate_every_n_blocks"]

# =============================================
# ۷. تنظیمات لاگ (Logging)
# =============================================
LOG_LEVEL = CONFIG["logging"]["level"]
LOG_FILE = os.path.join(OUTPUT_DIR, CONFIG["logging"]["log_file"])
LOG_TIMESTAMPS = CONFIG["logging"]["log_timestamps"]

# =============================================
# ۸. تنظیمات Benchmark
# =============================================
BENCHMARK_ENABLED = CONFIG["benchmark"]["enabled"]
BENCHMARK_RUN_ON_FIRST = CONFIG["benchmark"]["run_on_first_block"]
BENCHMARK_TEST_SIZES = CONFIG["benchmark"]["test_block_sizes"]

# =============================================
# ۹. ثابت‌های عددی و نوع داده
# =============================================
INT_DTYPE = np.int32
FLOAT_DTYPE = np.float32

# =============================================
# ۱۰. مقادیر معتبر برای best_dist
# =============================================
VALID_BEST_DIST = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# =============================================
# ۱۱. حداقل تعداد مقادیر معتبر برای برازش
# =============================================
MIN_VALID_VALUES = 10

# =============================================
# ۱۲. حداکثر تعداد مقادیر برای برازش (برای window cache)
# =============================================
MAX_VALUES_PER_FIT = 155  # 5 سال × 31 روز = 155

# =============================================
# ۱۳. تعریف توزیع‌ها (برای استفاده در zarr_schema و distributions)
# =============================================
DISTRIBUTIONS = [
    {
        "name": "normal",
        "params": [
            ("mean", "float32", "mean"),
            ("std", "float32", "std"),
        ],
        "fit_func": "fit_normal",
    },
    {
        "name": "lognormal",
        "params": [
            ("shape", "float32", "shape"),
            ("loc", "float32", "loc"),
            ("scale", "float32", "scale"),
        ],
        "fit_func": "fit_lognormal",
    },
    {
        "name": "gamma",
        "params": [
            ("alpha", "float32", "alpha"),
            ("loc", "float32", "loc"),
            ("beta", "float32", "beta"),
        ],
        "fit_func": "fit_gamma",
    },
    {
        "name": "weibull",
        "params": [
            ("shape", "float32", "shape"),
            ("loc", "float32", "loc"),
            ("scale", "float32", "scale"),
        ],
        "fit_func": "fit_weibull",
    },
]

# =============================================
# ۱۴. اعتبارسنجی تنظیمات
# =============================================
def validate_constants():
    errors = []
    if N_DAYS not in [365, 366]:
        errors.append(f"N_DAYS باید 365 یا 366 باشد (فعلاً {N_DAYS})")
    if VAR_INDEX_FOR_FIT not in range(N_VARS):
        errors.append(f"VAR_INDEX_FOR_FIT باید بین 0 تا {N_VARS-1} باشد (فعلاً {VAR_INDEX_FOR_FIT})")
    if not os.path.exists(CALENDAR_FILE):
        errors.append(f"فایل تقویم وجود ندارد: {CALENDAR_FILE}")
    if errors:
        raise ValueError("خطا در تنظیمات:\n" + "\n".join(errors))
    return True

validate_constants()

# =============================================
# نمایش اطلاعات (اختیاری)
# =============================================
if __name__ == "__main__":
    print(f"✅ تنظیمات بارگذاری شد:")
    print(f"   سال‌ها: {YEAR_START} تا {YEAR_END} ({N_YEARS} سال)")
    print(f"   روزها: {N_DAYS}")
    print(f"   متغیرها: {VARS} ({N_VARS} متغیر)")
    print(f"   پنجره: ±{WINDOW_DAYS} روز")
    print(f"   بلوک‌ها: {BLOCK_SIZE} ایستگاه")
    print(f"   MIN_VALID_VALUES: {MIN_VALID_VALUES}")
    print(f"   MAX_VALUES_PER_FIT: {MAX_VALUES_PER_FIT}")
    print(f"   تعداد توزیع‌ها: {len(DISTRIBUTIONS)}")