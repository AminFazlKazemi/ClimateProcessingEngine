import xarray as xr
import numpy as np
import time
from tqdm import tqdm

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
EXPECTED_DAYS = 366
EXPECTED_POINTS = 338627
SAMPLE_SIZE = 100000  # تعداد نمونه‌های تصادفی برای بررسی NaN (در هر متغیر)

print("📂 باز کردن فایل Zarr...")
ds = xr.open_zarr(ZARR_PATH, consolidated=False)
print(f"📊 تعداد متغیرها: {len(ds.data_vars)}")
print(f"📍 ابعاد مورد انتظار: ({EXPECTED_DAYS}, {EXPECTED_POINTS})")
print("=" * 90)

results = []
all_consistent = True
has_nan = False

# لیست متغیرهایی که به بررسی دقیق‌تر نیاز دارند
best_dist_vars = [v for v in ds.data_vars if v.endswith("_best_dist")]
count_vars = [v for v in ds.data_vars if v.endswith("_count")]

# تولید اندیس‌های تصادفی برای نمونه‌برداری
rng = np.random.default_rng(seed=42)
total_cells = EXPECTED_DAYS * EXPECTED_POINTS
sample_indices = rng.choice(total_cells, size=min(SAMPLE_SIZE, total_cells), replace=False)
sample_days = sample_indices // EXPECTED_POINTS
sample_points = sample_indices % EXPECTED_POINTS

for var_name in tqdm(ds.data_vars, desc="بررسی متغیرها"):
    arr = ds[var_name]
    shape = arr.shape
    status = "✅ کامل"
    details = ""

    # ۱. بررسی ابعاد
    if shape != (EXPECTED_DAYS, EXPECTED_POINTS):
        status = "❌ عدم تطابق ابعاد"
        details = f"shape={shape}"
        all_consistent = False
        results.append([var_name, shape, status, details])
        continue

    # ۲. بررسی محتوا با نمونه‌برداری (برای متغیرهای خاص)
    if var_name in best_dist_vars:
        # برای best_dist، نمونه‌برداری تصادفی انجام بده
        sampled = arr.isel(day=xr.DataArray(sample_days, dims="sample"),
                           point=xr.DataArray(sample_points, dims="sample")).values
        failed_count = np.sum(sampled == -1)
        invalid_count = np.sum((sampled < -1) | (sampled > 4))
        nan_count = np.sum(np.isnan(sampled))
        if nan_count > 0:
            status = "⚠️ دارای NaN"
            details = f"NaN نمونه={nan_count}/{SAMPLE_SIZE}"
            has_nan = True
        elif invalid_count > 0:
            status = "⚠️ مقدار نامعتبر"
            details = f"invalid نمونه={invalid_count}/{SAMPLE_SIZE}"
            all_consistent = False
        else:
            details = f"failed نمونه={failed_count}/{SAMPLE_SIZE} (برآورد {failed_count/SAMPLE_SIZE*100:.1f}%)"
            if failed_count > 0:
                status = "⚠️ برازش‌های ناموفق"
            else:
                status = "✅ کامل"

    elif var_name in count_vars:
        sampled = arr.isel(day=xr.DataArray(sample_days, dims="sample"),
                           point=xr.DataArray(sample_points, dims="sample")).values
        negative_count = np.sum(sampled < 0)
        nan_count = np.sum(np.isnan(sampled))
        if nan_count > 0:
            status = "⚠️ دارای NaN"
            details = f"NaN نمونه={nan_count}/{SAMPLE_SIZE}"
            has_nan = True
        elif negative_count > 0:
            status = "❌ دارای مقدار منفی"
            details = f"negative نمونه={negative_count}/{SAMPLE_SIZE}"
            all_consistent = False
        else:
            details = f"min نمونه={int(np.min(sampled))}, max نمونه={int(np.max(sampled))}"
            status = "✅ کامل"

    else:
        # سایر متغیرها (float): نمونه‌برداری تصادفی
        sampled = arr.isel(day=xr.DataArray(sample_days, dims="sample"),
                           point=xr.DataArray(sample_points, dims="sample")).values
        nan_count = np.sum(np.isnan(sampled))
        if nan_count > 0:
            status = "⚠️ دارای NaN"
            details = f"NaN نمونه={nan_count}/{SAMPLE_SIZE}"
            has_nan = True
        else:
            min_val = np.nanmin(sampled)
            max_val = np.nanmax(sampled)
            details = f"range نمونه=[{min_val:.2f}, {max_val:.2f}]"
            status = "✅ کامل"

    results.append([var_name, shape, status, details])

ds.close()

# ============================================================
# گزارش نهایی (ساده شده)
# ============================================================
print("\n" + "=" * 90)
print("📋 خلاصه وضعیت متغیرها (بر اساس نمونه‌برداری تصادفی)")
print("=" * 90)

# شمارش وضعیت‌ها
status_counts = {}
for _, _, status, _ in results:
    main_status = status.split()[0]
    status_counts[main_status] = status_counts.get(main_status, 0) + 1

for s, count in status_counts.items():
    emoji = {"✅": "✅", "❌": "❌", "⚠️": "⚠️"}.get(s, "•")
    print(f"   {emoji} {s}: {count} متغیر")

# نمایش متغیرهای مشکل‌دار
problematic = [r for r in results if r[2].startswith("❌") or r[2].startswith("⚠️")]
if problematic:
    print("\n⚠️ متغیرهای نیازمند بررسی:")
    for var_name, shape, status, details in problematic:
        print(f"   {var_name:35s} {status:20s} {details}")
else:
    print("\n🎉 **همه متغیرها از نظر ابعاد و نمونه‌برداری سالم به نظر می‌رسند.**")

print("=" * 90)