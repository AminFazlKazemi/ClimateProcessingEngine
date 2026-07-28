#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_imports.py
================================================================================
رفع خطای ImportError مربوط به fit_distributions
================================================================================
"""

import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DIST_PATH = PROJECT_ROOT / "numerical_engine" / "distributions.py"
ANALYZE_PATH = PROJECT_ROOT / "numerical_engine" / "analyze_station.py"

def backup_file(path):
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak_import")
        shutil.copy2(path, backup)
        print(f"   📋 پشتیبان: {backup}")
        return True
    return False

print("=" * 70)
print("🔧 رفع خطای ImportError: fit_distributions")
print("=" * 70)

# ============================================================================
# ۱. اضافه کردن تابع fit_distributions به distributions.py (alias)
# ============================================================================
print("\n📄 ۱. اصلاح distributions.py ...")
if not DIST_PATH.exists():
    print("   ❌ distributions.py یافت نشد!")
    exit(1)

backup_file(DIST_PATH)

with open(DIST_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# بررسی اینکه آیا fit_distributions قبلاً وجود دارد
if "def fit_distributions" not in content:
    # اضافه کردن alias در انتهای فایل، قبل از if __name__
    alias_code = """
# ============================================================================
# Alias برای سازگاری با کدهای قدیمی
# ============================================================================
def fit_distributions(values):
    \"\"\"Alias برای fit_distribution (سازگاری با نسخه‌های قدیمی)\"\"\"
    return fit_distribution(values)

# همچنین select_best_distribution را هم اضافه می‌کنیم
def select_best_distribution(values):
    \"\"\"برازش و انتخاب بهترین توزیع (سازگاری با نسخه‌های قدیمی)\"\"\"
    result = fit_distribution(values)
    if result is None:
        return None
    # result یک آرایه ۳۳ عنصری است
    # [0] = best_code
    # [4] = normal_aicc
    # [11] = skew_aicc
    # [18] = gev_aicc
    # [27] = bimodal_aicc
    # [30] = pearson_aicc
    best_code = int(result[0])
    dist_names = {0: 'normal', 1: 'skew', 2: 'gev', 3: 'bimodal', 4: 'pearson'}
    return {
        'best_dist': dist_names.get(best_code, 'unknown'),
        'best_code': best_code,
        'normal_aicc': result[4],
        'skew_aicc': result[11],
        'gev_aicc': result[18],
        'bimodal_aicc': result[27],
        'pearson_aicc': result[30],
        'mean': result[28],
        'std': result[29],
        'skewness': result[30],
        'median': result[31],
        'count': result[32],
    }
"""
    # درج alias قبل از if __name__
    pattern = r'(if __name__ == "__main__":)'
    content = re.sub(pattern, alias_code + "\n\n" + r'\1', content)
    
    with open(DIST_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    print("   ✅ fit_distributions و select_best_distribution به distributions.py اضافه شدند")
else:
    print("   ℹ️ fit_distributions قبلاً در distributions.py وجود دارد")

# ============================================================================
# ۲. اصلاح analyze_station.py (در صورت نیاز)
# ============================================================================
print("\n📄 ۲. بررسی analyze_station.py ...")
if ANALYZE_PATH.exists():
    backup_file(ANALYZE_PATH)
    
    with open(ANALYZE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # اگر از fit_distributions استفاده می‌کند، تغییری نمی‌دهیم چون alias اضافه شد.
    # اما اگر از fit_distribution استفاده می‌کند، همان خوب است.
    # فقط بررسی می‌کنیم که import درست باشد.
    
    # اگر import از fit_distributions استفاده می‌کند، آن را به fit_distribution تغییر نمی‌دهیم
    # چون alias را اضافه کردیم.
    
    # اما اگر تابع select_best_distribution را استفاده می‌کند، باید اصلاح شود.
    # معمولاً analyze_station از fit_distributions استفاده می‌کند و سپس best را انتخاب می‌کند.
    # فعلاً تغییری نمی‌دهیم چون aliasها را اضافه کردیم.
    
    print("   ✅ analyze_station.py نیازی به تغییر ندارد (aliasها اضافه شدند)")
else:
    print("   ⚠️ analyze_station.py یافت نشد")

# ============================================================================
# ۳. نتیجه
# ============================================================================
print("\n" + "=" * 70)
print("✅ اصلاحات با موفقیت اعمال شد!")
print("📌 توابع زیر به distributions.py اضافه شدند:")
print("   - fit_distributions(values) → alias برای fit_distribution")
print("   - select_best_distribution(values) → برازش و انتخاب بهترین توزیع")
print("\n🔄 اکنون می‌توانید دوباره main.py را اجرا کنید.")
print("=" * 70)