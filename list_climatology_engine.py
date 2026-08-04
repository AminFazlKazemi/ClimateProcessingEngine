#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
list_climatology_engine.py - نمایش ساختار و محتوای پوشه climatology_engine
"""

import os
from pathlib import Path

SOURCE = Path(r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine")

print("=" * 70)
print(f"📂 محتوای پوشه: {SOURCE}")
print("=" * 70)

# لیست همه فایل‌ها و پوشه‌ها (به جز پوشه‌های خیلی بزرگ)
total_files = 0
total_dirs = 0
ignored = 0

# پوشه‌هایی که نباید نمایش داده شوند
ignore_dirs = {
    "__pycache__", ".git", ".vscode", ".idea", "logs", "backup", "cache",
    "nature_output", "outlier_reports", ".pytest_cache", ".mypy_cache",
    "dist", "build", "htmlcov", ".coverage",
}

for root, dirs, files in os.walk(SOURCE):
    # حذف پوشه‌های ناخواسته از پیمایش
    dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith("backup_")]
    
    rel_root = Path(root).relative_to(SOURCE)
    if str(rel_root) == ".":
        print("\n📁 فایل‌های ریشه:")
    else:
        print(f"\n📁 {rel_root}/:")
    
    for file in sorted(files):
        if file.endswith((".pyc", ".pkl", ".pkl.gz", ".npy", ".log", ".tmp", ".zarr", ".nc")):
            ignored += 1
            continue
        if file in {"checkpoint.csv", "climatology.log", ".DS_Store", "Thumbs.db"}:
            ignored += 1
            continue
        file_path = Path(root) / file
        size_kb = file_path.stat().st_size / 1024
        print(f"   📄 {file} ({size_kb:.1f} KB)")
        total_files += 1
    
    total_dirs += 1

print("\n" + "=" * 70)
print(f"📊 خلاصه:")
print(f"   تعداد فایل‌های نمایش داده شده: {total_files}")
print(f"   تعداد فایل‌های نادیده گرفته شده: {ignored}")
print(f"   تعداد پوشه‌ها: {total_dirs}")
print("=" * 70)