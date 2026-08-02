#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reset_and_optimize.py
=========================================
۱. پاک کردن checkpoint.csv
۲. بهینه‌سازی config.yaml برای ساخت کش سریع
۳. اجرای build_cache_only.py
=========================================
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

# ============================================================
# مسیرها
# ============================================================
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.csv"

print("=" * 70)
print("🔄 ریست checkpoint و بهینه‌سازی config.yaml")
print("=" * 70)

# ============================================================
# ۱. پاک کردن checkpoint
# ============================================================
if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()
    print("✅ checkpoint.csv پاک شد.")
else:
    print("ℹ️ checkpoint.csv وجود نداشت.")

# ============================================================
# ۲. بهینه‌سازی config.yaml
# ============================================================
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # تنظیمات بهینه برای ساخت کش سریع
    config["block_size"] = 2000
    config["parallel"]["max_workers"] = 6
    config["use_parallel"] = True
    config["generate_percentile_maps"] = False  # غیرفعال برای سرعت
    config["validate_after_load"] = False
    config["validate_before_write"] = False
    config["validate_every_n_blocks"] = 10

    # اطمینان از اینکه cache فعال است
    if "paths" not in config:
        config["paths"] = {}
    config["paths"]["cache_dir"] = "./cache"

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print("✅ config.yaml بهینه‌سازی شد.")
    print(f"   block_size: 2000")
    print(f"   max_workers: 6")
    print(f"   generate_percentile_maps: False")
else:
    print("❌ config.yaml پیدا نشد!")
    sys.exit(1)

# ============================================================
# ۳. اجرای build_cache_only.py
# ============================================================
print("\n🚀 اجرای build_cache_only.py...")
print("   (فقط بلوک‌های ۰ تا ۱۹ پردازش می‌شوند)")
print("=" * 70)

result = subprocess.run(
    [sys.executable, str(BASE_DIR / "build_cache_only.py")],
    cwd=str(BASE_DIR)
)

if result.returncode == 0:
    print("\n✅ ساخت کش با موفقیت کامل شد!")
    print("📌 حالا main.py را اجرا کنید تا داده‌ها در Zarr نوشته شوند.")
else:
    print(f"\n❌ خطا در اجرا (کد بازگشت: {result.returncode})")