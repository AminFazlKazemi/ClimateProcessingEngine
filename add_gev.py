# -*- coding: utf-8 -*-
"""
add_gev.py - اضافه کردن توزیع GEV به مجموعه توزیع‌ها (5 توزیع)
"""

import re
from pathlib import Path

BASE_DIR = Path(__file__).parent


def add_gev_to_distributions():
    """افزودن تابع fit_gev و به‌روزرسانی fit_distributions"""
    file_path = BASE_DIR / "numerical_engine" / "distributions.py"
    if not file_path.exists():
        print(f"⚠️ فایل {file_path} یافت نشد.")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # اگر GEV قبلاً اضافه شده، خروج
    if "fit_gev" in content and "gev" in content:
        print("ℹ️ GEV قبلاً در distributions.py اضافه شده است.")
        return True

    # اضافه کردن تابع fit_gev
    gev_func = '''

def fit_gev(data):
    """
    برازش توزیع مقادیر حدی تعمیم‌یافته (GEV)
    پارامترها: (c, loc, scale) که c = shape, loc = location, scale = scale
    """
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan,
                "aic": np.nan, "bic": np.nan, "loglik": np.nan}
    try:
        params = stats.genextreme.fit(data)
        c, loc, scale = params[0], params[1], params[2]
        loglik = np.sum(stats.genextreme.logpdf(data, c, loc, scale))
        n = len(data)
        k = 3
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        return {"p1": loc, "p2": scale, "p3": c,
                "aic": aic, "bic": bic, "loglik": loglik}
    except Exception:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan,
                "aic": np.nan, "bic": np.nan, "loglik": np.nan}
'''

    # پیدا کردن محل درج (بعد از fit_bimodal یا قبل از fit_distributions)
    if "def fit_distributions" in content:
        # درج قبل از fit_distributions
        pattern = r'(def fit_distributions\(data\):)'
        replacement = gev_func + '\n\n' + r'\1'
        content = re.sub(pattern, replacement, content, count=1)
    else:
        # اگر fit_distributions پیدا نشد، در انتها اضافه کن
        content += '\n' + gev_func

    # به‌روزرسانی fit_distributions برای اضافه کردن "gev"
    pattern = r'(fits\s*=\s*\{[\s\S]*?)(\})'
    def add_gev_to_fits(match):
        body = match.group(1)
        if '"gev":' not in body:
            # اضافه کردن GEV به دیکشنری
            # پیدا کردن آخرین آیتم قبل از }
            body = body.rstrip()
            if body.endswith(','):
                body += '\n        "gev": fit_gev(data)'
            else:
                body += ',\n        "gev": fit_gev(data)'
        return body + '\n    }'

    content = re.sub(pattern, add_gev_to_fits, content, flags=re.DOTALL)

    # اضافه کردن import stats.genextreme اگر وجود ندارد
    if "from scipy.stats import genextreme" not in content:
        # بعد از import های موجود اضافه کن
        content = content.replace(
            "from scipy import stats",
            "from scipy import stats\nfrom scipy.stats import genextreme"
        )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ distributions.py به‌روز شد (GEV اضافه شد).")
    return True


def add_gev_to_analyze_station():
    """به‌روزرسانی analyze_station.py برای پشتیبانی از GEV"""
    file_path = BASE_DIR / "numerical_engine" / "analyze_station.py"
    if not file_path.exists():
        print(f"⚠️ فایل {file_path} یافت نشد.")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # اضافه کردن DIST_MAP اگر وجود ندارد
    if "DIST_MAP" not in content:
        # اضافه کردن قبل از تابع analyze_station
        dist_map = '''
# نگاشت نام توزیع به عدد
DIST_MAP = {"normal": 0, "pearson": 1, "skewnormal": 2, "bimodal": 3, "gev": 4}
'''
        # درج بعد از import ها
        import_pattern = r'(import numpy as np.*?from numerical_engine\.window_engine import compute_windows)'
        if re.search(import_pattern, content, re.DOTALL):
            content = re.sub(import_pattern, r'\1\n' + dist_map, content, flags=re.DOTALL)
        else:
            # اگر الگو پیدا نشد، در ابتدای فایل اضافه کن
            content = dist_map + content
    else:
        # به‌روزرسانی DIST_MAP
        if "gev" not in content:
            content = content.replace(
                'DIST_MAP = {"normal": 0, "pearson": 1, "skewnormal": 2, "bimodal": 3}',
                'DIST_MAP = {"normal": 0, "pearson": 1, "skewnormal": 2, "bimodal": 3, "gev": 4}'
            )

    # به‌روزرسانی dist_names در analyze_station
    if "dist_names = [" in content:
        pattern = r'dist_names\s*=\s*\[.*?\]'
        if '"gev"' not in content:
            new_names = 'dist_names = ["normal", "pearson", "skewnormal", "bimodal", "gev"]'
            content = re.sub(pattern, new_names, content)

    # به‌روزرسانی _empty_result
    if "_empty_result" in content:
        pattern = r'(dist_names\s*=\s*\[.*?\])'
        if '"gev"' not in re.search(pattern, content).group(0):
            content = re.sub(
                pattern,
                'dist_names = ["normal", "pearson", "skewnormal", "bimodal", "gev"]',
                content
            )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ analyze_station.py به‌روز شد (GEV اضافه شد).")
    return True


def add_gev_to_write_block():
    """به‌روزرسانی write_block.py برای ساخت VAR_NAMES داینامیک با GEV"""
    file_path = BASE_DIR / "result_pipeline" / "write_block.py"
    if not file_path.exists():
        print(f"⚠️ فایل {file_path} یافت نشد.")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # اگر VAR_NAMES به‌صورت لیست ثابت تعریف شده، آن را با نسخه داینامیک جایگزین کن
    if "VAR_NAMES = [" in content:
        # پیدا کردن محل تعریف VAR_NAMES
        if "dist_names" not in content:
            new_var_names = '''
# لیست نام توزیع‌ها (برای ساخت داینامیک VAR_NAMES)
dist_names = ["normal", "pearson", "skewnormal", "bimodal", "gev"]

# ساخت VAR_NAMES به‌صورت داینامیک
VAR_NAMES = []
for var in ["tmax", "tmean", "tmin"]:
    # آمار پایه
    for stat in ["count", "mean", "std", "median", "min", "max", "skewness"]:
        VAR_NAMES.append(f"{var}_{stat}")
    # پارامترهای توزیع‌ها
    for dist in dist_names:
        for p in range(1, 6):
            VAR_NAMES.append(f"{var}_{dist}_p{p}")
    # معیارهای انتخاب توزیع
    VAR_NAMES.append(f"{var}_best_dist")
    VAR_NAMES.append(f"{var}_aic")
    VAR_NAMES.append(f"{var}_bic")
    VAR_NAMES.append(f"{var}_loglik")
'''
            # جایگزینی کل بخش VAR_NAMES
            pattern = r'VAR_NAMES\s*=\s*\[.*?\]'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                content = content.replace(match.group(0), new_var_names.strip())
            else:
                # اگر VAR_NAMES پیدا نشد، در انتهای فایل اضافه کن
                content += '\n' + new_var_names
        else:
            # اگر dist_names قبلاً وجود دارد، فقط "gev" را اضافه کن
            if "gev" not in content:
                content = content.replace(
                    'dist_names = ["normal", "pearson", "skewnormal", "bimodal"]',
                    'dist_names = ["normal", "pearson", "skewnormal", "bimodal", "gev"]'
                )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ write_block.py به‌روز شد (VAR_NAMES با GEV به‌روز شد).")
    return True


def main():
    print("🚀 اضافه کردن توزیع GEV به پروژه...")
    ok1 = add_gev_to_distributions()
    ok2 = add_gev_to_analyze_station()
    ok3 = add_gev_to_write_block()
    if ok1 and ok2 and ok3:
        print("✅ تمام تغییرات با موفقیت اعمال شدند.")
        print("📊 اکنون 5 توزیع (Normal, Pearson, Skew-Normal, Bimodal, GEV) در دسترس هستند.")
    else:
        print("⚠️ برخی تغییرات اعمال نشدند. لطفاً خطاها را بررسی کنید.")


if __name__ == "__main__":
    main()