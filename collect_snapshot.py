#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_snapshot.py
================================================================================
اسکریپت جمع‌آوری اسنپ‌شات کامل از کل پروژه (مستقل از اصلاحات آینده)
برای ذخیره‌سازی در گیت‌هاب، ارسال به دیگران، یا بک‌آپ
================================================================================
نحوه اجرا:
    python collect_snapshot.py

خروجی:
    project_snapshot.txt  (در ریشه پروژه)
================================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# تنظیمات
# ============================================================================

# ریشه پروژه (همان جایی که اسکریپت در آن قرار دارد)
PROJECT_ROOT = Path(__file__).parent.absolute()
OUTPUT_FILE = PROJECT_ROOT / "project_snapshot.txt"

# پسوندهای قابل قبول برای اسنپ‌شات
EXTENSIONS = (
    ".py", ".yaml", ".yml", ".json", ".txt", ".md", ".cff",
    ".ini", ".cfg", ".sh", ".bat", ".ps1", ".gitignore",
    ".dockerignore", ".editorconfig", ".pre-commit-config.yaml"
)

# پوشه‌هایی که نباید اسکن شوند
EXCLUDE_DIRS = {
    "__pycache__", ".git", ".vscode", ".idea", "dist", "build",
    "*.egg-info", "htmlcov", ".pytest_cache", ".mypy_cache",
    "venv", "env", "conda-env", ".venv", "node_modules",
    "backup_*", "*.zarr", "*.nc", "*.hdf5", "*.parquet",
}

# نام فایل‌هایی که نباید اسکن شوند
EXCLUDE_FILES = {
    "project_snapshot.txt",  # خود خروجی
    "collect_snapshot.py",   # خود اسکریپت (اختیاری - می‌تواند شامل شود)
}

# ============================================================================
# توابع
# ============================================================================

def should_include(path: Path) -> bool:
    """
    بررسی اینکه آیا فایل باید در اسنپ‌شات گنجانده شود
    """
    # نادیده گرفتن پوشه‌های خاص
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
        # پشتیبانی از wildcard ساده
        if part.startswith("backup_") and part.endswith("_"):
            return False

    # نادیده گرفتن فایل‌های خاص
    if path.name in EXCLUDE_FILES:
        return False

    # نادیده گرفتن فایل‌های پشتیبان
    if path.name.endswith((".bak", ".bak2", ".bak3", "~", ".tmp")):
        return False

    # فقط پسوندهای مشخص
    if path.suffix.lower() not in EXTENSIONS:
        return False

    # فایل‌های خیلی بزرگ را نادیده بگیر (بیش از ۵ مگابایت)
    if path.stat().st_size > 5 * 1024 * 1024:
        return False

    return True


def collect_files() -> str:
    """
    پیمایش تمام فایل‌های پروژه و جمع‌آوری محتوا
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"📁 SNAPSHOT از پروژه climatology_engine")
    lines.append(f"📂 ریشه: {PROJECT_ROOT}")
    lines.append(f"📅 تاریخ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")

    total_files = 0

    # پیمایش بازگشتی از ریشه پروژه
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)

        # حذف پوشه‌های ناخواسته از پیمایش (برای بهبود سرعت)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("backup_")]

        for file in sorted(files):
            file_path = root_path / file
            rel_path = file_path.relative_to(PROJECT_ROOT)

            if not should_include(file_path):
                continue

            total_files += 1

            # خواندن محتوای فایل
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # فایل باینری (مثلاً .pyc) - فقط نام را ثبت کن
                content = f"⚠️ فایل باینری (قابل نمایش نیست): {file_path.suffix}"
            except Exception as e:
                content = f"⚠️ خطا در خواندن فایل: {e}"

            lines.append(f"\n{'=' * 80}")
            lines.append(f"📄 فایل: {rel_path}")
            lines.append(f"{'=' * 80}")
            lines.append(content)
            lines.append("")  # خط خالی بین فایل‌ها

    # آمار نهایی
    lines.append("\n" + "=" * 80)
    lines.append(f"📊 آمار اسنپ‌شات:")
    lines.append(f"   تعداد فایل‌های اسکن‌شده: {total_files}")
    lines.append(f"   تاریخ ایجاد: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# اجرای اصلی
# ============================================================================

if __name__ == "__main__":
    print(f"📂 جمع‌آوری اسنپ‌شات از: {PROJECT_ROOT}")
    print("⏳ در حال پردازش...")

    snapshot = collect_files()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(snapshot)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"✅ فایل اسنپ‌شات در {OUTPUT_FILE} ذخیره شد.")
    print(f"📊 حجم فایل: {size_kb:.1f} کیلوبایت")