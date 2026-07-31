# -*- coding: utf-8 -*-
"""
constants.py – automatically generated with safe defaults.
"""

import os
import yaml
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

CONFIG = load_config(CONFIG_PATH)

# =============================================
# ۱. تنظیمات سال و روز
# =============================================
YEAR_START = CONFIG.get("years", {}).get("start", 1370)
YEAR_END = CONFIG.get("years", {}).get("end", 1399)
YEAR_LIST = list(range(YEAR_START, YEAR_END + 1))
N_YEARS = len(YEAR_LIST)
N_DAYS = CONFIG.get("days", 366)

# =============================================
# ۲. تنظیمات متغیرها
# =============================================
VARS = CONFIG.get("variables", ["tmin", "tmean", "tmax"])
N_VARS = len(VARS)
VAR_INDEX = {name: idx for idx, name in enumerate(VARS)}
VAR_INDEX_FOR_FIT = CONFIG.get("fit_variable_index", 1)

# =============================================
# ۳. تنظیمات پنجره (Window)
# =============================================
WINDOW_DAYS = CONFIG.get("window_days", 2)
WINDOW_SIZE = 2 * WINDOW_DAYS + 1
WINDOW_TYPE = CONFIG.get("window_type", "centered")
MIN_VALID_YEARS = CONFIG.get("min_valid_years", 10)

# =============================================
# ۴. تنظیمات مسیرها (Paths) – با اولویت config.yaml
# =============================================
PATHS = CONFIG.get("paths", {})

INPUT_ZARR_BASE = PATHS.get("input_zarr_base", r"K:\gozareshha\dr vazife\140504 - qc temp\zarr_yearly_monthly")
ZARR_BASE = PATHS.get("zarr_base", INPUT_ZARR_BASE)

OUTPUT_DIR = PATHS.get("output_dir", r"./nature_output")
OUTPUT_ZARR_NAME = PATHS.get("output_zarr_name", r"climatology_stationwise_final.zarr")
OUTPUT_ZARR = os.path.join(OUTPUT_DIR, OUTPUT_ZARR_NAME)

# ============================================================
# ✅ اصلاح مهم: مسیر مطلق checkpoint
# ============================================================
CHECKPOINT_FILE = PATHS.get("checkpoint_file", "checkpoint.csv")
# اگر مسیر نسبی است، آن را به OUTPUT_DIR متصل کن
if not os.path.isabs(CHECKPOINT_FILE):
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, CHECKPOINT_FILE)
CHECKPOINT_PATH = CHECKPOINT_FILE  # برای سازگاری

# بقیه‌ی مسیرها
LOG_FILE = PATHS.get("log_file", r"climatology.log")
CACHE_DIR = PATHS.get("cache_dir", r"./cache")
SAMPLE_DATA_DIR = PATHS.get("sample_data_dir", r"./sample_data")
CALENDAR_FILE = PATHS.get("calendar_file", r"K:/Temp/needed/calendar.txt")

# =============================================
# ۵. تنظیمات پردازش
# =============================================
USE_PARALLEL = CONFIG.get("use_parallel", True)
BLOCK_SIZE = CONFIG.get("block_size", 30000)
CHUNK_SIZE = CONFIG.get("chunk_size", 500)
CORES = CONFIG.get("cores", 0)

MAX_BLOCKS_IN_MEMORY = CONFIG.get("processing", {}).get("max_blocks_in_memory", 5)
CHUNK_SIZE_ZARR = CONFIG.get("processing", {}).get("chunk_size", [366, 500])
COMPRESSION = CONFIG.get("processing", {}).get("compression", "zstd")
COMPRESSION_LEVEL = CONFIG.get("processing", {}).get("compression_level", 3)
N_POINTS_MAX = CONFIG.get("processing", {}).get("n_points_max", 40000)
OUTPUT_PRECISION = CONFIG.get("processing", {}).get("output_precision", "float32")

# =============================================
# ۶. تنظیمات اعتبارسنجی (Validation)
# =============================================
VALIDATE_AFTER_LOAD = CONFIG.get("validate_after_load", False)
VALIDATE_BEFORE_WRITE = CONFIG.get("validate_before_write", False)
VALIDATE_EVERY_N_BLOCKS = CONFIG.get("validate_every_n_blocks", 10)

# =============================================
# ۷. تنظیمات لاگ
# =============================================
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "INFO")

# =============================================
# ۸. داده‌های اضافی
# =============================================
DATA_FORMAT = CONFIG.get("data_format", "auto")
POINT_SAMPLING = CONFIG.get("point_sampling", "all")
N_SAMPLE_POINTS = CONFIG.get("n_sample_points", 40000)

# =============================================
# ۹. انواع داده‌ها
# =============================================
FLOAT_DTYPE = np.float32 if OUTPUT_PRECISION == "float32" else np.float64
INT_DTYPE = np.int32

# =============================================
# ۱۰. تنظیمات اضافی (اختیاری)
# =============================================
BOOTSTRAP_ENABLED = CONFIG.get("bootstrap", {}).get("enabled", True)
BOOTSTRAP_ITERATIONS = CONFIG.get("bootstrap", {}).get("n_iterations", 100)
BOOTSTRAP_CONFIDENCE = CONFIG.get("bootstrap", {}).get("confidence_level", 0.95)
BOOTSTRAP_SEED = CONFIG.get("bootstrap", {}).get("random_seed", 42)

QUALITY_MIN_SAMPLE = CONFIG.get("quality", {}).get("min_sample_size", 3)
QUALITY_THRESHOLD_AICC = CONFIG.get("quality", {}).get("threshold_aicc", 1000)
QUALITY_THRESHOLD_SKEW = CONFIG.get("quality", {}).get("threshold_skew", 5.0)
QUALITY_DETECT_OUTLIERS = CONFIG.get("quality", {}).get("detect_outliers", False)
QUALITY_OUTLIER_SIGMA = CONFIG.get("quality", {}).get("outlier_sigma", 4.0)

# =============================================
# ۱۱. توزیع‌های فعال
# =============================================
DISTRIBUTIONS = CONFIG.get("distributions", {})
NORMAL_MODE_DISTS = DISTRIBUTIONS.get("normal_mode", ["normal", "skew", "bimodal", "pearson"])
EXTREME_MODE_DISTS = DISTRIBUTIONS.get("extreme_mode", ["normal", "skew", "gev", "bimodal", "pearson"])

# =============================================
# ۱۲. تنظیمات خروجی
# =============================================
OUTPUT_FORMAT = CONFIG.get("output", {}).get("format", "zarr")
OUTPUT_OVERWRITE = CONFIG.get("output", {}).get("overwrite", True)
OUTPUT_INCLUDE_METADATA = CONFIG.get("output", {}).get("include_metadata", True)
OUTPUT_INCLUDE_PROVENANCE = CONFIG.get("output", {}).get("include_provenance", True)

# =============================================
# ۱۳. تنظیمات بصری‌سازی
# =============================================
VIZ_ENABLED = CONFIG.get("visualization", {}).get("enabled", True)
VIZ_DIR = CONFIG.get("visualization", {}).get("output_dir", "./visualizations")
VIZ_FORMATS = CONFIG.get("visualization", {}).get("format", ["png", "pdf"])
VIZ_INTERACTIVE = CONFIG.get("visualization", {}).get("interactive", True)
VIZ_MAP_PROJECTION = CONFIG.get("visualization", {}).get("map_projection", "PlateCarree")
VIZ_DPI = CONFIG.get("visualization", {}).get("dpi", 300)

LOG_TIMESTAMPS = True

# =============================================
# توزیع‌های معتبر
# =============================================
VALID_BEST_DIST = {-1, 0, 1, 2, 3, 4}   # ۴ = GEV

# =============================================
# پارامترهای محاسباتی
# =============================================
MAX_VALUES_PER_FIT = N_YEARS * WINDOW_SIZE
MIN_VALID_VALUES = 5

# =============================================
# برای سازگاری با کدهای قدیمی
# =============================================
CHECKPOINT_PATH = CHECKPOINT_FILE