#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_cache_coverage.py – بررسی محدوده‌ی نقاط کش‌شده
- چند فایل کش را نمونه‌برداری می‌کند
- شکل و محدوده‌ی نقاط را نشان می‌دهد
- تخمین می‌زند که کش تا چه نقطه‌ای ساخته شده
"""

import os
import sys
import pickle
import gzip
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path("cache")
if not CACHE_DIR.exists():
    print("❌ پوشه‌ی کش وجود ندارد.")
    sys.exit(1)

files = list(CACHE_DIR.glob("*.pkl.gz")) + list(CACHE_DIR.glob("*.pkl"))
print(f"📂 تعداد کل فایل‌های کش: {len(files):,}")

# ============================================================================
# نمونه‌برداری از ۱۰ فایل اول (و چند فایل تصادفی)
# ============================================================================
sample_files = files[:10]  # ۱۰ فایل اول

# اضافه کردن چند فایل تصادفی (اگر بیشتر از ۱۰ تا باشد)
if len(files) > 10:
    import random
    random.seed(42)
    sample_files.extend(random.sample(files[10:], min(20, len(files) - 10)))

print(f"🔍 بررسی {len(sample_files)} فایل نمونه...")

# ============================================================================
# بررسی هر فایل نمونه
# ============================================================================
block_info = defaultdict(int)  # شمارش تعداد فایل‌ها برای هر محدوده

for f in sample_files:
    try:
        if f.suffix == '.gz':
            with gzip.open(f, 'rb') as fp:
                data = pickle.load(fp)
        else:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
        
        # تشخیص محدوده‌ی نقاط از روی شکل داده
        if hasattr(data, 'shape') and len(data.shape) >= 2:
            # شکل داده معمولاً (days, block_size) است
            # یا (days, block_size, vars)
            if data.shape[1] <= 2000:
                block_size = data.shape[1]
            else:
                block_size = data.shape[0] if data.shape[0] <= 2000 else None
            
            if block_size:
                # تخمین block_start از نام فایل ممکن نیست، اما از محتوا هم نمی‌شود.
                # فقط block_size قابل تشخیص است.
                block_info[f"block_size_{block_size}"] += 1
                print(f"   ✅ {f.name}: block_size={block_size}, shape={data.shape}")
            else:
                print(f"   ⚠️ {f.name}: shape غیرمنتظره: {data.shape}")
        else:
            print(f"   ⚠️ {f.name}: داده قابل تشخیص نیست (type={type(data)})")
    except Exception as e:
        print(f"   ❌ {f.name}: خطا در خواندن: {e}")

# ============================================================================
# نتیجه‌گیری
# ============================================================================
print("\n" + "=" * 80)
print("📊 نتیجه‌گیری:")

if block_info:
    print("   block_size‌های پیدا شده در کش:")
    for key, count in block_info.items():
        print(f"      {key}: {count} فایل")

    # اگر همه‌ی block_size‌ها ۱۰۰۰ یا ۲۰۰۰ باشند، یعنی کش برای بلوک‌های قبلی ساخته شده
    if all('block_size_1000' in k or 'block_size_2000' in k for k in block_info.keys()):
        print("\n   ✅ کش فعلی برای بلوک‌های **۴۰,۰۰۰ نقطه** (با block_size ۱۰۰۰ یا ۲۰۰۰) ساخته شده.")
        print("   ⚠️ برای ۳۳۸,۶۲۷ نقطه، با block_size جدید، کش جدید ساخته می‌شود.")
    else:
        print("\n   ⚠️ کش شامل block_sizeهای مختلف است. ممکن است ترکیبی از کش‌های قدیم و جدید باشد.")
else:
    print("   ❌ هیچ داده‌ای قابل تشخیص از فایل‌های نمونه پیدا نشد.")

print("=" * 80)