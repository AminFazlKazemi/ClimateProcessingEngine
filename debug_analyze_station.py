# -*- coding: utf-8 -*-
"""
debug_analyze_station.py - دیباگ و رفع خطای تعداد آرگومان‌های analyze_station
"""

import re
import inspect
from pathlib import Path

BASE_DIR = Path(__file__).parent


def analyze_function_signature(file_path, func_name):
    """تحلیل امضای تابع از فایل"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # جستجوی تعریف تابع
    pattern = rf'def\s+{func_name}\s*\(([^)]*)\):'
    match = re.search(pattern, content)
    if not match:
        return None, None
    
    params = match.group(1).strip()
    # استخراج نام پارامترها
    param_list = [p.strip().split(':')[0].split('=')[0].strip() 
                  for p in params.split(',') if p.strip()]
    return param_list, content


def fix_analyze_station_signature():
    """اصلاح امضای analyze_station و فراخوانی آن"""
    
    # 1. بررسی فایل analyze_station.py
    ana_path = BASE_DIR / "numerical_engine" / "analyze_station.py"
    if not ana_path.exists():
        print(f"❌ فایل {ana_path} یافت نشد!")
        return False
    
    params, content = analyze_function_signature(ana_path, "analyze_station")
    print(f"📄 analyze_station.py امضای فعلی: analyze_station({', '.join(params) if params else '...'})")
    
    # 2. بررسی فایل process_block.py
    proc_path = BASE_DIR / "orchestrator" / "process_block.py"
    if not proc_path.exists():
        print(f"❌ فایل {proc_path} یافت نشد!")
        return False
    
    with open(proc_path, 'r', encoding='utf-8') as f:
        proc_content = f.read()
    
    # جستجوی فراخوانی analyze_station
    call_pattern = r'analyze_station\s*\(([^)]*)\)'
    calls = re.findall(call_pattern, proc_content)
    if calls:
        print(f"📄 process_block.py فراخوانی‌های analyze_station:")
        for i, call in enumerate(calls):
            args = [a.strip() for a in call.split(',')]
            print(f"   فراخوانی {i+1}: analyze_station({', '.join(args)})")
    
    # 3. تشخیص مشکل
    if params is None:
        print("❌ تابع analyze_station در فایل پیدا نشد!")
        return False
    
    # اگر تعداد پارامترها ۳ باشد (بدون window_table)
    if len(params) == 3 and 'window_table' not in params:
        print("⚠️ تابع analyze_station فقط ۳ پارامتر دارد (بدون window_table)")
        print("   اما در process_block با ۴ آرگومان فراخوانی شده است.")
        
        # اصلاح: اضافه کردن window_table به امضای تابع
        print("🔧 در حال اصلاح analyze_station.py برای پذیرش window_table...")
        
        # یافتن خط def
        def_pattern = r'(def analyze_station\([^)]*\):)'
        new_def = 'def analyze_station(station_data, year_list, window_table, var_idx):'
        content = re.sub(def_pattern, new_def, content, count=1)
        
        # همچنین باید در بدنه تابع، از window_table استفاده شود
        # اگر window_table در بدنه استفاده نشده، آن را به عنوان پارامتر اضافی نادیده می‌گیریم
        # اما برای اطمینان، یک کامنت اضافه می‌کنیم
        if 'window_table' not in content:
            # اضافه کردن یک خط که window_table را نادیده می‌گیرد (برای جلوگیری از خطای استفاده نشده)
            content = content.replace(
                'def analyze_station(station_data, year_list, window_table, var_idx):',
                'def analyze_station(station_data, year_list, window_table, var_idx):\n    # window_table برای استفاده‌های آینده نگهداری می‌شود'
            )
        
        with open(ana_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ analyze_station.py اصلاح شد (window_table اضافه شد).")
        return True
    
    # اگر تعداد پارامترها ۴ باشد اما نام‌ها متفاوت است
    elif len(params) == 4:
        print("✅ analyze_station دارای ۴ پارامتر است، اما ممکن است ترتیب یا نام‌ها متفاوت باشد.")
        # بررسی فراخوانی در process_block
        if calls:
            call_args = [a.strip() for a in calls[0].split(',') if a.strip()]
            if len(call_args) == 4:
                print("✅ فراخوانی با ۴ آرگومان انجام شده است.")
                # بررسی تطابق نام‌ها
                if set(params) == set(['station_data', 'year_list', 'window_table', 'var_idx']):
                    print("✅ نام پارامترها با فراخوانی مطابقت دارد.")
                    return True
                else:
                    print("⚠️ نام پارامترها با فراخوانی مطابقت ندارد.")
                    print(f"   پارامترهای تابع: {params}")
                    print(f"   آرگومان‌های فراخوانی: {call_args}")
                    # اصلاح: تغییر نام پارامترها در تابع
                    # این کار پیچیده است، بهتر است فراخوانی را اصلاح کنیم
                    # اما برای سادگی، فراخوانی را به ترتیب درست می‌کنیم
                    # ابتدا بررسی کنیم که آیا window_table در فراخوانی وجود دارد؟
                    if 'window_table' in call_args:
                        # فراخوانی درست است، فقط ترتیب را اصلاح می‌کنیم
                        # اما اگر نام‌ها متفاوت است، بهتر است تابع را اصلاح کنیم
                        # برای سادگی، فراخوانی را با نام‌های کلیدی بازنویسی می‌کنیم
                        new_call = "analyze_station(station_data=station_data, year_list=year_list, window_table=window_table, var_idx=var_idx)"
                        proc_content = proc_content.replace(calls[0], new_call)
                        with open(proc_path, 'w', encoding='utf-8') as f:
                            f.write(proc_content)
                        print("✅ process_block.py اصلاح شد (فراخوانی با نام‌های کلیدی).")
                        return True
    else:
        print(f"⚠️ تعداد پارامترهای analyze_station: {len(params)} (غیرمنتظره)")
        print("   نیاز به بررسی دستی دارد.")
        return False


def main():
    print("=" * 60)
    print("🔍 دیباگ خطای analyze_station")
    print("=" * 60)
    
    if fix_analyze_station_signature():
        print("\n✅ اصلاحات با موفقیت اعمال شد.")
        print("🔄 لطفاً دوباره main.py را اجرا کنید.")
    else:
        print("\n⚠️ اصلاحات انجام نشد. لطفاً دستی اقدام کنید:")
        print("   1. فایل numerical_engine/analyze_station.py را باز کنید.")
        print("   2. امضای تابع را به شکل زیر تغییر دهید:")
        print("      def analyze_station(station_data, year_list, window_table, var_idx):")
        print("   3. در orchestrator/process_block.py، فراخوانی را به شکل زیر اصلاح کنید:")
        print("      result = analyze_station(station_data, year_list, window_table, var_idx)")


if __name__ == "__main__":
    main()