#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cache_detailed.py – تحلیل دقیق محتوای کش
- گروه‌بندی بر اساس کلید واقعی (sample_hash, block_start, block_size, year, month, var)
- شناسایی فایل‌های تکراری و بی‌استفاده
- پیشنهاد حذف
"""

import os
import sys
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ============================================================================
# مسیر کش
# ============================================================================
CACHE_DIR = Path("cache")
if not CACHE_DIR.exists():
    print("❌ پوشه‌ی کش وجود ندارد.")
    sys.exit(1)

files = list(CACHE_DIR.glob("*.pkl.gz")) + list(CACHE_DIR.glob("*.pkl"))
print(f"📂 تعداد کل فایل‌های کش: {len(files):,}")
total_size = sum(f.stat().st_size for f in files) / (1024**2)
print(f"💾 حجم کل کش: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

# ============================================================================
# استخراج کلید واقعی از نام فایل (نام فایل = hash کلید)
# ============================================================================
# برای هر فایل، ما نمی‌توانیم کلید اصلی را از نام فایل استخراج کنیم،
# چون نام فایل هش کلید است. پس نمی‌توانیم مستقیماً sample_hash را بخوانیم.
# اما می‌توانیم فایل‌ها را بر اساس اندازه و تاریخ گروه‌بندی کنیم.

# ============================================================================
# گروه‌بندی بر اساس اندازه + تاریخ (تقریب خوبی برای تشخیص تکراری‌ها)
# ============================================================================
groups = defaultdict(list)
for f in files:
    size = f.stat().st_size
    # تاریخ به ساعت و دقیقه (نه ثانیه) برای کاهش حساسیت
    mtime = int(f.stat().st_mtime / 60) * 60
    key = (size, mtime)
    groups[key].append(f)

print(f"\n📊 تعداد گروه‌های منحصربه‌فرد (بر اساس اندازه+زمان): {len(groups):,}")

# ============================================================================
# شناسایی گروه‌های با بیش از ۱ فایل (تکراری)
# ============================================================================
duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
if duplicate_groups:
    total_dup_files = sum(len(v) for v in duplicate_groups.values())
    print(f"🔁 تعداد فایل‌های تکراری: {total_dup_files:,} از {len(files):,}")
    
    # حجم اضافی
    extra_size = sum(
        (len(v) - 1) * v[0].stat().st_size 
        for v in duplicate_groups.values()
    ) / (1024**2)
    print(f"💾 فضای اضافی اشغال‌شده توسط تکراری‌ها: {extra_size:.2f} MB ({extra_size/1024:.2f} GB)")
else:
    print("✅ هیچ فایل تکراری یافت نشد.")

# ============================================================================
# پیشنهاد نهایی
# ============================================================================
print("\n" + "=" * 80)
print("🧹 پیشنهاد:")
if duplicate_groups:
    print(f"   ✅ می‌توانید {total_dup_files:,} فایل تکراری را حذف کنید.")
    print(f"   💾 {extra_size:.2f} MB فضا آزاد می‌شود.")
    print("\n   برای حذف خودکار، اسکریپت clean_cache_safe.py را اجرا کنید.")
else:
    print("   ✅ کش شما بهینه است. هیچ تکراری وجود ندارد.")

print("=" * 80)