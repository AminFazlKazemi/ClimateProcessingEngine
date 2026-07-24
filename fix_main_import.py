#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_main_import.py
================================================================================
رفع مستقیم خطای ImportError: N_OUTPUTS در main.py
این اسکریپت فقط فایل main.py را اصلاح می‌کند.
================================================================================
"""

import os
import re

# مسیر فایل main.py
MAIN_PATH = r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\main.py"

def fix_main():
    if not os.path.exists(MAIN_PATH):
        print(f"❌ فایل main.py در مسیر {MAIN_PATH} یافت نشد!")
        return False

    with open(MAIN_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # پشتیبان
    backup = MAIN_PATH + ".bak3"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📋 پشتیبان: {backup}")

    # الگوی import از constants
    pattern = r"(from constants import \(([^)]*)\))"

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("⚠️ الگوی import از constants پیدا نشد. شاید فایل قبلاً اصلاح شده است.")
        return False

    full_import = match.group(0)
    import_body = match.group(2)

    # جدا کردن خطوط import
    lines = [line.strip() for line in import_body.split(",") if line.strip()]
    
    # حذف N_OUTPUTS از لیست
    new_lines = [line for line in lines if "N_OUTPUTS" not in line]

    # ساخت import جدید
    new_import_body = ",\n    ".join(new_lines)
    new_import = f"from constants import (\n    {new_import_body}\n)"

    # جایگزینی در محتوا
    new_content = content.replace(full_import, new_import)

    # اضافه کردن import از zarr_schema اگر وجود ندارد
    if "from zarr_schema import N_OUTPUTS" not in new_content:
        # پیدا کردن آخرین import برای اضافه کردن بعد از آن
        last_import_pos = -1
        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("import") or line.strip().startswith("from"):
                last_import_pos = i
        if last_import_pos >= 0:
            lines.insert(last_import_pos + 1, "from zarr_schema import N_OUTPUTS")
        else:
            lines.insert(0, "from zarr_schema import N_OUTPUTS")
        new_content = "\n".join(lines)

    # ذخیره فایل
    with open(MAIN_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ main.py اصلاح شد.")
    return True

if __name__ == "__main__":
    success = fix_main()
    if success:
        print("\n🔄 حالا می‌توانید دوباره اجرا کنید:")
        print("   python main.py")
    else:
        print("\n⚠️ لطفاً مشکل را بررسی کنید.")