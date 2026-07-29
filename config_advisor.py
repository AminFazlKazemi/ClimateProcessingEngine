#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config_advisor.py – نسخه کامل با توضیحات بیشتر
================================================================================
ارزیابی خودکار config.yaml و ارائه راهنمای تنظیم بهینه
برای موتور اقلیم‌شناسی (ClimateProcessingEngine)
================================================================================
"""

import os
import sys
import yaml
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# تلاش برای import psutil (اختیاری)
# ============================================================================
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil نصب نیست. اطلاعات سخت‌افزاری کامل در دسترس نخواهد بود.")
    print("   برای نصب: pip install psutil\n")

# ============================================================================
# رنگ‌ها برای خروجی زیبا
# ============================================================================
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_colored(text, color=Colors.RESET, bold=False):
    """چاپ متن با رنگ در ترمینال"""
    if not sys.stdout.isatty():
        print(text)
        return
    prefix = Colors.BOLD if bold else ""
    print(f"{prefix}{color}{text}{Colors.RESET}")

# ============================================================================
# ۱. خواندن و اعتبارسنجی config.yaml
# ============================================================================
def load_config(config_path):
    """
    بارگذاری فایل YAML و بازگرداندن دیکشنری
    """
    if not os.path.exists(config_path):
        print_colored(f"❌ فایل {config_path} یافت نشد!", Colors.RED, bold=True)
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print_colored(f"✅ فایل {config_path} با موفقیت بارگذاری شد.", Colors.GREEN)
        return config
    except Exception as e:
        print_colored(f"❌ خطا در خواندن {config_path}: {e}", Colors.RED, bold=True)
        return None

# ============================================================================
# ۲. تشخیص سخت‌افزار
# ============================================================================
def get_hardware_info():
    """
    دریافت اطلاعات RAM، CPU، دیسک با استفاده از psutil
    """
    info = {}
    if HAS_PSUTIL:
        # اطلاعات حافظه
        mem = psutil.virtual_memory()
        info['ram_total_gb'] = mem.total / (1024**3)
        info['ram_available_gb'] = mem.available / (1024**3)
        info['ram_percent'] = mem.percent
        
        # اطلاعات CPU
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_physical'] = psutil.cpu_count(logical=False)
        
        # اطلاعات دیسک
        disk_usage = {}
        for partition in psutil.disk_partitions():
            if partition.fstype and 'cdrom' not in partition.opts:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        'total_gb': usage.total / (1024**3),
                        'free_gb': usage.free / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'percent': usage.percent
                    }
                except Exception:
                    pass
        info['disk_usage'] = disk_usage
    else:
        info['ram_total_gb'] = None
        info['ram_available_gb'] = None
        info['ram_percent'] = None
        info['cpu_count'] = None
        info['cpu_physical'] = None
        info['disk_usage'] = {}
    return info

# ============================================================================
# ۳. بررسی مسیرها و فایل‌های ورودی
# ============================================================================
def validate_paths(config):
    """
    بررسی وجود مسیرهای ورودی و خروجی
    """
    issues = []
    warnings_list = []
    
    # مسیر ورودی Zarr
    input_base = config.get('paths', {}).get('input_zarr_base') or config.get('paths', {}).get('zarr_base')
    if input_base:
        if os.path.exists(input_base):
            zarr_files = glob.glob(os.path.join(input_base, "*.zarr"))
            if zarr_files:
                issues.append(f"✅ ورودی Zarr: {len(zarr_files)} فایل در {input_base}")
            else:
                warnings_list.append(f"⚠️ هیچ فایل Zarr در {input_base} یافت نشد!")
        else:
            warnings_list.append(f"❌ مسیر ورودی Zarr وجود ندارد: {input_base}")
    
    # فایل calendar.txt
    calendar_file = config.get('paths', {}).get('calendar_file')
    if calendar_file:
        if os.path.exists(calendar_file):
            issues.append(f"✅ calendar.txt: {calendar_file} (موجود)")
        else:
            warnings_list.append(f"⚠️ فایل calendar.txt در {calendar_file} یافت نشد!")
    
    # دایرکتوری خروجی
    output_dir = config.get('paths', {}).get('output_dir')
    if output_dir:
        if os.path.exists(output_dir):
            issues.append(f"✅ دایرکتوری خروجی: {output_dir} (موجود)")
        else:
            warnings_list.append(f"⚠️ دایرکتوری خروجی وجود ندارد: {output_dir} (ایجاد خواهد شد)")
    
    # فایل checkpoint
    checkpoint_file = config.get('paths', {}).get('checkpoint_file')
    if checkpoint_file and os.path.exists(checkpoint_file):
        warnings_list.append(f"⚠️ فایل checkpoint موجود است: {checkpoint_file} (در صورت نیاز حذف کنید)")
    
    return issues, warnings_list

# ============================================================================
# ۴. ارزیابی پارامترهای حیاتی بر اساس سخت‌افزار
# ============================================================================
def evaluate_parameters(config, hardware):
    """
    مقایسه مقادیر فعلی با مقادیر پیشنهادی
    """
    recommendations = []
    warnings_list = []
    
    # ۴-۱. block_size و max_workers
    block_size = config.get('block_size', 5000)
    max_workers = config.get('parallel', {}).get('max_workers', 14)
    ram_gb = hardware.get('ram_total_gb', 32)  # fallback به 32
    
    # محاسبه حافظه مصرفی هر بلوک (تقریبی)
    # block_size × 31 (years) × 366 (days) × 3 (vars) × 4 (bytes)
    ram_per_block_mb = (block_size * 31 * 366 * 3 * 4) / (1024**2)
    total_ram_estimate_gb = (ram_per_block_mb * max_workers) / 1024 + 4  # +4 GB سیستم
    
    if ram_gb:
        if total_ram_estimate_gb > ram_gb * 0.8:
            warnings_list.append(
                f"⚠️ حافظهٔ تخمینی {total_ram_estimate_gb:.1f} GB بیش از ۸۰٪ RAM موجود ({ram_gb:.1f} GB) است!"
            )
            if block_size > 1000:
                recommendations.append(f"🔧 block_size را از {block_size} به ۲۰۰۰ یا کمتر کاهش دهید")
            if max_workers > 6:
                recommendations.append(f"🔧 max_workers را از {max_workers} به ۴ یا ۶ کاهش دهید")
        elif total_ram_estimate_gb > ram_gb * 0.6:
            warnings_list.append(
                f"⚠️ حافظهٔ تخمینی {total_ram_estimate_gb:.1f} GB حدود {total_ram_estimate_gb/ram_gb*100:.0f}٪ RAM است (ترجیحاً زیر ۶۰٪)"
            )
        else:
            recommendations.append(f"✅ تنظیمات حافظه مناسب است (~{total_ram_estimate_gb/ram_gb*100:.0f}٪ RAM)")
    
    # ۴-۲. پیشنهاد block_size بر اساس RAM
    if ram_gb:
        if ram_gb < 16:
            recommended_block_size = 1000
        elif ram_gb < 32:
            recommended_block_size = 1500
        elif ram_gb < 64:
            recommended_block_size = 2000
        else:
            recommended_block_size = 5000
        
        if block_size != recommended_block_size:
            recommendations.append(
                f"🔧 block_size پیشنهادی برای RAM {ram_gb:.1f} GB: {recommended_block_size} (فعلی: {block_size})"
            )
    
    # ۴-۳. max_workers
    cpu_count = hardware.get('cpu_count', 8)
    if cpu_count:
        recommended_workers = min(cpu_count - 1, 8) if cpu_count > 2 else 2
        if max_workers > cpu_count:
            warnings_list.append(f"⚠️ max_workers ({max_workers}) بیشتر از تعداد هسته‌های CPU ({cpu_count}) است!")
            recommendations.append(f"🔧 max_workers را حداکثر به {cpu_count} کاهش دهید")
        elif max_workers > recommended_workers + 2:
            recommendations.append(f"🔧 max_workers پیشنهادی: {recommended_workers} (فعلی: {max_workers})")
    
    # ۴-۴. n_points_max
    n_points_max = config.get('processing', {}).get('n_points_max', 40000)
    n_sample = config.get('n_sample_points', 40000)
    if n_points_max == 40000 and n_sample == 40000:
        # اگر هر دو ۴۰۰۰۰ باشند، ممکن است کاربر نخواهد همه نقاط را پردازش کند
        recommendations.append(
            "ℹ️ n_points_max و n_sample_points هر دو ۴۰۰۰۰ هستند. اگر همه نقاط را می‌خواهید، مقدار را به تعداد کل نقاط افزایش دهید."
        )
    elif n_points_max < n_sample:
        warnings_list.append(
            f"⚠️ n_points_max ({n_points_max}) کمتر از n_sample_points ({n_sample}) است. نقاط نمونه‌گیری بیشتر از نقاط پردازشی است."
        )
    
    # ۴-۵. max_blocks_in_memory
    max_blocks = config.get('processing', {}).get('max_blocks_in_memory', 3)
    if ram_gb and ram_gb < 32 and max_blocks > 2:
        recommendations.append(f"🔧 max_blocks_in_memory را از {max_blocks} به ۲ کاهش دهید (برای RAM کمتر از ۳۲ GB)")
    
    # ۴-۶. output_precision
    precision = config.get('processing', {}).get('output_precision', 'float32')
    if precision != 'float32':
        recommendations.append(f"ℹ️ output_precision: {precision} (float32 برای صرفه‌جویی در حافظه و فضا پیشنهاد می‌شود)")
    
    # ۴-۷. compression
    compression = config.get('processing', {}).get('compression', 'zstd')
    if compression not in ['zstd', 'blosc']:
        recommendations.append(f"ℹ️ compression: {compression} (zstd یا blosc پیشنهاد می‌شود)")
    
    # ۴-۸. use_parallel
    if not config.get('use_parallel', True):
        recommendations.append("ℹ️ use_parallel: false (فعال کردن آن سرعت را افزایش می‌دهد)")
    
    return recommendations, warnings_list

# ============================================================================
# ۵. بررسی تنظیمات توزیع‌ها
# ============================================================================
def evaluate_distributions(config):
    """
    بررسی تنظیمات توزیع‌ها
    """
    issues = []
    dist_normal = config.get('distributions', {}).get('normal_mode', [])
    dist_extreme = config.get('distributions', {}).get('extreme_mode', [])
    
    if not dist_normal:
        issues.append("⚠️ normal_mode خالی است! حداقل Normal را اضافه کنید.")
    elif 'normal' not in dist_normal:
        issues.append("⚠️ normal_mode باید شامل 'normal' باشد (توزیع پایه).")
    
    if not dist_extreme:
        issues.append("⚠️ extreme_mode خالی است! حداقل Normal و GEV را اضافه کنید.")
    
    return issues

# ============================================================================
# ۶. بررسی اعتبارسنجی
# ============================================================================
def evaluate_validation(config):
    """
    بررسی تنظیمات اعتبارسنجی
    """
    issues = []
    validate_after = config.get('validate_after_load', False)
    validate_before = config.get('validate_before_write', False)
    validate_every = config.get('validate_every_n_blocks', 10)
    
    if validate_after:
        issues.append("ℹ️ validate_after_load: true (کند می‌کند، برای دیباگ مفید است)")
    if validate_before:
        issues.append("ℹ️ validate_before_write: true (کند می‌کند، برای دیباگ مفید است)")
    if validate_every < 5:
        issues.append("ℹ️ validate_every_n_blocks بسیار کوچک است (کند می‌شود). ۱۰ یا بیشتر پیشنهاد می‌شود.")
    
    return issues

# ============================================================================
# ۷. گزارش نهایی
# ============================================================================
def generate_report(config, hardware, issues, warnings_list, recommendations):
    """
    تولید گزارش نهایی
    """
    print("\n" + "=" * 80)
    print_colored("📊 گزارش ارزیابی config.yaml", Colors.CYAN, bold=True)
    print("=" * 80)
    
    # سخت‌افزار
    print("\n🖥️  اطلاعات سخت‌افزاری:")
    if hardware.get('ram_total_gb'):
        print(f"   RAM کل: {hardware['ram_total_gb']:.1f} GB")
        print(f"   RAM موجود: {hardware['ram_available_gb']:.1f} GB ({hardware['ram_percent']:.1f}% استفاده)")
    if hardware.get('cpu_count'):
        print(f"   CPU: {hardware['cpu_count']} هسته منطقی, {hardware['cpu_physical']} هسته فیزیکی")
    if hardware.get('disk_usage'):
        for mount, usage in hardware['disk_usage'].items():
            print(f"   دیسک {mount}: {usage['free_gb']:.1f} GB آزاد از {usage['total_gb']:.1f} GB")
    print()
    
    # مسیرها
    print("📂 بررسی مسیرها:")
    for item in issues:
        if item.startswith("✅"):
            print_colored(f"   {item}", Colors.GREEN)
        elif item.startswith("⚠️"):
            print_colored(f"   {item}", Colors.YELLOW)
        else:
            print(f"   {item}")
    
    for warn in warnings_list:
        if warn.startswith("⚠️"):
            print_colored(f"   {warn}", Colors.YELLOW)
        elif warn.startswith("❌"):
            print_colored(f"   {warn}", Colors.RED)
        else:
            print(f"   {warn}")
    
    # توصیه‌ها
    if recommendations:
        print("\n💡 توصیه‌های تنظیم:")
        for rec in recommendations:
            if rec.startswith("✅"):
                print_colored(f"   {rec}", Colors.GREEN)
            elif rec.startswith("🔧"):
                print_colored(f"   {rec}", Colors.BLUE)
            elif rec.startswith("⚠️"):
                print_colored(f"   {rec}", Colors.YELLOW)
            else:
                print(f"   {rec}")
    else:
        print_colored("\n✅ هیچ توصیه‌ای برای تغییر وجود ندارد. تنظیمات بهینه است.", Colors.GREEN)
    
    # خلاصه
    print("\n" + "=" * 80)
    total_issues = len([i for i in issues if not i.startswith("✅")]) + len(warnings_list) + len(recommendations)
    if total_issues == 0:
        print_colored("✅ همه چیز عالی است! می‌توانید main.py را اجرا کنید.", Colors.GREEN, bold=True)
    else:
        print_colored(f"⚠️ {total_issues} مورد نیاز به بررسی/تغییر دارد.", Colors.YELLOW, bold=True)
    print("=" * 80)
    
    # پیشنهادات نهایی
    print("\n📌 پیشنهادات نهایی:")
    print("   1. برای پردازش همه نقاط، n_points_max را به تعداد کل نقاط افزایش دهید.")
    print("   2. اگر RAM محدود است، block_size و max_workers را کاهش دهید.")
    print("   3. از float32 برای صرفه‌جویی در فضا استفاده کنید.")
    print("   4. checkpoint را در صورت وجود حذف کنید تا از ابتدا شروع شود.")
    print("   5. برای تست اولیه، n_points_max را به ۱۰۰۰ کاهش دهید.")

# ============================================================================
# ۸. تابع اصلی
# ============================================================================
def main():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    print_colored("🚀 راهنمای تنظیم خودکار config.yaml", Colors.CYAN, bold=True)
    print("=" * 80)
    
    # بارگذاری config
    config = load_config(config_path)
    if config is None:
        return
    
    # دریافت اطلاعات سخت‌افزاری
    hardware = get_hardware_info()
    
    # اعتبارسنجی مسیرها
    path_issues, path_warnings = validate_paths(config)
    
    # ارزیابی پارامترها
    recs, warns = evaluate_parameters(config, hardware)
    
    # ارزیابی توزیع‌ها
    dist_issues = evaluate_distributions(config)
    
    # ارزیابی اعتبارسنجی
    val_issues = evaluate_validation(config)
    
    # ترکیب همه موارد
    all_issues = path_issues + dist_issues + val_issues
    all_warnings = path_warnings + warns
    
    # گزارش نهایی
    generate_report(config, hardware, all_issues, all_warnings, recs)

if __name__ == "__main__":
    main()