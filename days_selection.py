import numpy as np
import pandas as pd

# ۱. محاسبه میانگین فضایی بارش برای هر روز
print("محاسبه میانگین فضایی بارش روزانه...")
daily_means = p_full.mean(dim=['latitude', 'longitude']).compute().values
n_days = len(daily_means)

# ۲. استخراج ماه و فصل از تاریخ‌های شمسی
times = p_full.time.values
months = np.array([int(str(t)[4:6]) for t in times])  # دو رقم میانی = ماه

def get_season(month):
    if month in [1, 2, 3]:   # فروردین، اردیبهشت، خرداد
        return 0  # بهار
    elif month in [4, 5, 6]: # تیر، مرداد، شهریور
        return 1  # تابستان
    elif month in [7, 8, 9]: # مهر، آبان، آذر
        return 2  # پاییز
    else:                    # دی، بهمن، اسفند
        return 3  # زمستان

seasons = np.array([get_season(m) for m in months])
season_names = ['بهار', 'تابستان', 'پاییز', 'زمستان']

# ۳. انتخاب روزهای نماینده برای هر فصل
selected_days = []
np.random.seed(42)  # برای تکرارپذیری

for season_idx in range(4):
    mask = (seasons == season_idx)
    if np.sum(mask) == 0:
        continue
    
    # مقادیر بارش برای این فصل
    season_means = daily_means[mask]
    season_indices = np.where(mask)[0]
    
    # مرتب‌سازی بر اساس بارش
    sorted_idx = np.argsort(season_means)
    sorted_means = season_means[sorted_idx]
    sorted_indices = season_indices[sorted_idx]
    
    n_season = len(sorted_means)
    
    # ۱. روز خشک (صدک ۲۰)
    dry_idx = sorted_indices[int(n_season * 0.20)]
    # ۲. روز متوسط (صدک ۵۰)
    mid_idx = sorted_indices[int(n_season * 0.50)]
    # ۳. روز مرطوب (صدک ۸۰)
    wet_idx = sorted_indices[int(n_season * 0.80)]
    
    selected_days.extend([dry_idx, mid_idx, wet_idx])
    
    # چاپ اطلاعات
    print(f"\nفصل {season_names[season_idx]}:")
    print(f"  روز خشک (صدک ۲۰): day={dry_idx}, بارش={daily_means[dry_idx]:.3f} mm")
    print(f"  روز متوسط (صدک ۵۰): day={mid_idx}, بارش={daily_means[mid_idx]:.3f} mm")
    print(f"  روز مرطوب (صدک ۸۰): day={wet_idx}, بارش={daily_means[wet_idx]:.3f} mm")

# مرتب‌سازی نهایی بر اساس زمان
selected_days = sorted(set(selected_days))
print(f"\n✅ {len(selected_days)} روز نماینده انتخاب شدند:")
print(selected_days)