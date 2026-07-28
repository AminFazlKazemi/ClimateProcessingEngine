#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_main_syntax.py
================================================================================
رفع خطای SyntaxError در main.py (جابجایی فراخوانی warmup_numba)
================================================================================
"""

import re
import shutil
from pathlib import Path

MAIN_PATH = Path("main.py")
if not MAIN_PATH.exists():
    print("❌ main.py یافت نشد!")
    exit(1)

# پشتیبان
BACKUP = MAIN_PATH.with_suffix(".py.bak_syntax")
shutil.copy2(MAIN_PATH, BACKUP)
print(f"📋 پشتیبان: {BACKUP}")

with open(MAIN_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ۱. حذف فراخوانی warmup_numba() که ممکن است اشتباه درج شده باشد
# الگو: یک خط با warmup_numba() که ممکن است تورفتگی اشتباه داشته باشد
# ما کل بلوک warmup را حذف می‌کنیم و دوباره به‌صورت درست اضافه می‌کنیم.

# ابتدا هرگونه فراخوانی warmup_numba() را حذف می‌کنیم
content = re.sub(r'\n\s*warmup_numba\(\)\s*\n', '\n', content)

# ۲. تعریف تابع warmup_numba را اگر وجود داشت، حذف می‌کنیم (چون قبلاً اضافه شده)
# ولی ممکن است قبلاً اضافه شده باشد. اگر وجود داشت، آن را نگه می‌داریم.
# فقط اگر تابع warmup_numba تعریف نشده بود، آن را اضافه می‌کنیم.
# اما با توجه به خروجی قبلی، warmup قبلاً به main.py اضافه شده است، پس فقط فراخوانی مشکل داشت.

# ۳. فراخوانی صحیح را در ابتدای تابع main، قبل از هر try اضافه می‌کنیم
# الگوی "def main():" را پیدا می‌کنیم و بعد از آن یک خط جدید اضافه می‌کنیم.
pattern = r'(def main\(\):.*?)(?=\n    try:)'
# اما بهتر است بعد از خط "def main():" و قبل از هر کد دیگر، warmup را اضافه کنیم.

# روش ساده‌تر: بعد از خط "def main():" یک خط جدید با تورفتگی مناسب اضافه کنیم.
# تورفتگی تابع main ۴ فاصله است، پس warmup را با ۴ فاصله تورفتگی می‌دهیم.
warmup_call = "\n    # Warmup Numba (pre-compile JIT)\n    warmup_numba()\n"

# پیدا کردن موقعیت def main()
match = re.search(r'def main\(\):', content)
if match:
    pos = match.end()
    # درج بعد از آن
    content = content[:pos] + warmup_call + content[pos:]
else:
    print("⚠️ تابع main() پیدا نشد!")

# ۴. ذخیره فایل
with open(MAIN_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ main.py اصلاح شد. فراخوانی warmup به ابتدای تابع main منتقل شد.")
print("🔄 اکنون می‌توانید دوباره main.py را اجرا کنید.")