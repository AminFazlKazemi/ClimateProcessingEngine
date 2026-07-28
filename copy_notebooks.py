#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
copy_notebooks.py - کپی فایل‌های نوت‌بوک از source به target
"""

import os
import shutil
from pathlib import Path

# ============================================================
# مسیرها
# ============================================================
SOURCE = Path(r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine")
TARGET = Path(r"K:\gozareshha\dr vazife\140504 - qc temp\ClimateProcessingEngine")

# ============================================================
# پوشه‌ی notebooks در مقصد
# ============================================================
TARGET_NOTEBOOKS = TARGET / "notebooks"
TARGET_NOTEBOOKS.mkdir(parents=True, exist_ok=True)

# ============================================================
# پیدا کردن و کپی فایل‌های .ipynb
# ============================================================
source_notebooks = SOURCE / "notebooks"
if not source_notebooks.exists():
    print(f"❌ پوشه‌ی notebooks در source پیدا نشد: {source_notebooks}")
    exit(1)

# لیست تمام فایل‌های .ipynb
notebook_files = list(source_notebooks.glob("*.ipynb"))
print(f"📂 تعداد فایل‌های نوت‌بوک در source: {len(notebook_files)}")

copied = 0
for src_file in notebook_files:
    dst_file = TARGET_NOTEBOOKS / src_file.name
    try:
        shutil.copy2(src_file, dst_file)
        print(f"✅ کپی شد: {src_file.name}")
        copied += 1
    except Exception as e:
        print(f"❌ خطا در کپی {src_file.name}: {e}")

print(f"\n✅ تعداد کل فایل‌های کپی‌شده: {copied}")

# ============================================================
# (اختیاری) بررسی اینکه آیا فایل‌های دیگری در target موجودند که در source نیستند
# ============================================================
target_files = set(TARGET_NOTEBOOKS.glob("*.ipynb"))
source_names = {f.name for f in notebook_files}
extra = [f for f in target_files if f.name not in source_names]
if extra:
    print("\n⚠️ فایل‌های زیر در target هستند اما در source وجود ندارند:")
    for f in extra:
        print(f"   🗑️ {f.name}")
    answer = input("❓ آیا می‌خواهید آن‌ها را حذف کنید؟ (y/n): ").strip().lower()
    if answer == "y":
        for f in extra:
            f.unlink()
            print(f"🗑️ حذف شد: {f.name}")
    else:
        print("⏭️ حذف نشد.")
else:
    print("✅ همه فایل‌های target با source هماهنگ هستند.")