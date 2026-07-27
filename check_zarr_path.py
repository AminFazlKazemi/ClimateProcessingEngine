import os
import glob

ZARR_BASE_CONFIG = r"K:/gozareshha/dr vazife/140504 - qc temp/zarr_yearly_monthly"

print(f"🔍 بررسی مسیر: {ZARR_BASE_CONFIG}")

if os.path.exists(ZARR_BASE_CONFIG):
    print("✅ مسیر وجود دارد.")
    zarr_files = glob.glob(os.path.join(ZARR_BASE_CONFIG, "*.zarr"))
    print(f"   تعداد فایل‌های Zarr پیدا شده: {len(zarr_files)}")
    if len(zarr_files) > 0:
        print("   نمونه فایل‌ها:")
        for f in zarr_files[:5]:
            print(f"      - {os.path.basename(f)}")
        if len(zarr_files) > 5:
            print(f"      ... و {len(zarr_files) - 5} فایل دیگر")
    else:
        print("⚠️ هیچ فایل Zarr در این مسیر پیدا نشد!")
        print("   (چرا که اسکریپت منتظر داده‌های واقعی است و داده‌ساختگی تولید نمی‌کند)")
else:
    print("❌ مسیر وجود ندارد!")
    print("   لطفاً مسیر صحیح Zarr را در config.yaml تنظیم کنید.")