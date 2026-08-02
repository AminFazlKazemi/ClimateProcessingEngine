import zarr
from constants import N_DAYS

ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
TARGET_POINTS = 338627  # تعداد کل نقاطی که باید باشد

print("📂 در حال باز کردن Zarr...")
root = zarr.open(ZARR_PATH, mode='a')

all_arrays = list(root.array_keys())
print(f"🔍 تعداد کل متغیرها: {len(all_arrays)}")

for name in all_arrays:
    arr = root[name]
    current_shape = arr.shape
    if current_shape[1] != TARGET_POINTS:
        print(f"   📏 تغییر اندازه: {name} از {current_shape} به (366, {TARGET_POINTS})")
        arr.resize((N_DAYS, TARGET_POINTS))
    else:
        print(f"   ✅ {name} در حال حاضر سایز صحیح دارد: {current_shape}")

print("✅ همه متغیرها هم‌اندازه شدند.")