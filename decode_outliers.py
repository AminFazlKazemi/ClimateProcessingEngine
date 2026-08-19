import pandas as pd
import numpy as np

# مسیرها
OUTLIER_CSV = r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\outlier_reports\outliers.csv"
METADATA_FILE = r"M:\New folder1\needed\xy-elevation.txt"

print("📂 در حال بارگذاری فایل خام پرت‌ها...")
df = pd.read_csv(OUTLIER_CSV)

print("📂 در حال بارگذاری فایل متادیتا (xy-elevation.txt) برای دریافت کد واقعی ایستگاه‌ها...")
# فرض: فایل xy-elevation.txt با فاصله یا تب جدا شده است (با توجه به نمونه)
metadata_df = pd.read_csv(METADATA_FILE, sep=r'\s+')

# استخراج stationid از فایل متادیتا (به همان ترتیب)
station_ids_real = metadata_df['stationid'].values
print(f"✅ {len(station_ids_real)} ایستگاه در فایل متادیتا پیدا شد.")

# ======================================================================
# ۱. جایگزینی station_idx با کد واقعی (stationid) با استفاده از xy-elevation.txt
# ======================================================================
# station_idx های 0 تا N-1 را به stationid های همان ایندکس در فایل متادیتا نگاشت می‌کنیم
df['real_stationid'] = df['station_idx'].apply(lambda x: station_ids_real[int(x)])

# ======================================================================
# ۲. اصلاح دما (تقسیم بر ۱۰)
# ======================================================================
df['corrected_value'] = df['value'] / 10.0

print("\n📊 نمونه‌ای از ۱۰ رکورد اول با داده‌های اصلاح‌شده (با stationid واقعی):")
cols = ['real_stationid', 'day', 'corrected_value', 'var']
print(df[cols].head(10))

# ============================================================
# (اختیاری) ذخیره نسخه اصلاح‌شده در یک فایل CSV جدید
# ============================================================
df_corrected = df[['real_stationid', 'day', 'corrected_value', 'var', 'timestamp']]
df_corrected.to_csv(r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\outlier_reports\outliers_corrected.csv", index=False)
print("💾 فایل اصلاح‌شده ذخیره شد.")
print(r"K:\gozareshha\dr vazife\140504 - qc temp\climatology\climatology_engine\outlier_reports\outliers_corrected.csv")