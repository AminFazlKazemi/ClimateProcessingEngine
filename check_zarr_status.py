#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_zarr_status.py – بررسی کامل بودن بلوک‌های Zarr خروجی
================================================================================
بررسی می‌کند که آیا بلوک‌های قبلی (قبل از بلوک جاری) به‌درستی کامل شده‌اند.
نشان می‌دهد که هر متغیر در هر بلوک چند نقطه معتبر دارد.
================================================================================
"""

import os
import numpy as np
import xarray as xr

# ============================================================================
# تنظیمات
# ============================================================================
ZARR_PATH = r"I:/climatology_366_rolling/climatology_stationwise_final.zarr"
BLOCK_SIZE = 2000
VARS = ['tmin', 'tmean', 'tmax']
DAY_INDEX = 0  # روز اول (برای بررسی کامل بودن)

print("=" * 80)
print("🔍 بررسی وضعیت بلوک‌های Zarr")
print(f"   فایل: {ZARR_PATH}")
print(f"   اندازه بلوک: {BLOCK_SIZE}")
print("=" * 80)

# ============================================================================
# ۱. باز کردن Zarr
# ============================================================================
if not os.path.exists(ZARR_PATH):
    print(f"❌ فایل Zarr وجود ندارد: {ZARR_PATH}")
    exit(1)

ds = xr.open_zarr(ZARR_PATH, consolidated=False)
n_stations = ds.sizes['point']
n_days = ds.sizes['day']
print(f"\n📊 ابعاد: {n_days} روز × {n_stations:,} ایستگاه")

# ============================================================================
# ۲. بررسی هر متغیر
# ============================================================================
total_blocks = (n_stations + BLOCK_SIZE - 1) // BLOCK_SIZE
print(f"\n📦 تعداد کل بلوک‌ها: {total_blocks}")

# ذخیره وضعیت هر بلوک برای هر متغیر
block_status = {var: [] for var in VARS}
complete_blocks = {var: 0 for var in VARS}
incomplete_blocks = {var: 0 for var in VARS}

for var in VARS:
    mean_var = f'{var}_mean'
    if mean_var not in ds:
        print(f"\n⚠️ متغیر {mean_var} در Zarr وجود ندارد!")
        continue
    
    data = ds[mean_var].isel(day=DAY_INDEX).values
    print(f"\n🔹 {var.upper()}:")
    
    for block_idx in range(total_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, n_stations)
        block_data = data[start:end]
        valid_count = np.sum(~np.isnan(block_data))
        total_count = len(block_data)
        is_complete = (valid_count == total_count)
        
        block_status[var].append({
            'block': block_idx,
            'start': start,
            'end': end,
            'valid': valid_count,
            'total': total_count,
            'complete': is_complete
        })
        
        if is_complete:
            complete_blocks[var] += 1
        else:
            incomplete_blocks[var] += 1
    
    print(f"   بلوک‌های کامل: {complete_blocks[var]} از {total_blocks}")
    print(f"   بلوک‌های ناقص: {incomplete_blocks[var]} از {total_blocks}")
    
    # نمایش اولین بلوک ناقص
    for block_info in block_status[var]:
        if not block_info['complete']:
            print(f"   ⚠️ اولین بلوک ناقص: بلوک {block_info['block']} "
                  f"(ایستگاه‌های {block_info['start']:,}-{block_info['end']:,}) "
                  f"→ {block_info['valid']:,}/{block_info['total']:,} معتبر")
            break
    else:
        print(f"   ✅ همه بلوک‌ها کامل هستند!")

# ============================================================================
# ۳. پیدا کردن آخرین بلوک مشترک کامل (بین هر سه متغیر)
# ============================================================================
print("\n" + "=" * 80)
print("📌 آخرین بلوکی که هر سه متغیر در آن کامل هستند:")

min_complete = total_blocks
for var in VARS:
    # تعداد بلوک‌های کامل این متغیر
    complete = sum(1 for b in block_status[var] if b['complete'])
    if complete < min_complete:
        min_complete = complete

# اگر min_complete > 0 باشد، یعنی بلوک‌های ۰ تا min_complete-1 برای همه متغیرها کامل هستند.
# اما ممکن است یک بلوک خاص برای یک متغیر کامل نباشد.
# بهتر است به‌جای تعداد، آخرین اندیس بلوکی که همه متغیرها در آن کامل هستند را پیدا کنیم.
last_common_complete = -1
for block_idx in range(total_blocks):
    all_complete = True
    for var in VARS:
        if block_idx >= len(block_status[var]):
            all_complete = False
            break
        if not block_status[var][block_idx]['complete']:
            all_complete = False
            break
    if all_complete:
        last_common_complete = block_idx
    else:
        # اولین بلوک ناقص را پیدا کردیم، پس بلوک قبل آن آخرین بلوک مشترک کامل است
        break

if last_common_complete >= 0:
    print(f"   ✅ بلوک‌های ۰ تا {last_common_complete} برای همه متغیرها کامل هستند.")
    print(f"   📍 اولین بلوک ناقص مشترک: بلوک {last_common_complete + 1}")
else:
    print("   ⚠️ هیچ بلوک مشترک کاملی یافت نشد!")

# ============================================================================
# ۴. خلاصه نهایی
# ============================================================================
print("\n" + "=" * 80)
print("📊 خلاصه نهایی:")
print(f"   تعداد کل بلوک‌ها: {total_blocks}")
print(f"   آخرین بلوک مشترک کامل: {last_common_complete}")
print(f"   بلوک‌های باقی‌مانده برای پردازش: {total_blocks - last_common_complete - 1}")

# نمایش وضعیت هر متغیر در بلوک جاری (اگر مشخص است)
current_block = 67  # از لاگ: block_start=134000 → block_idx=67
if current_block < total_blocks:
    print(f"\n🔍 وضعیت بلوک جاری (بلوک {current_block}):")
    for var in VARS:
        if current_block < len(block_status[var]):
            info = block_status[var][current_block]
            status = "✅ کامل" if info['complete'] else f"⚠️ {info['valid']:,}/{info['total']:,}"
            print(f"   {var.upper()}: {status}")

# ============================================================================
# ۵. بررسی نهایی
# ============================================================================
if last_common_complete >= 0:
    print("\n" + "=" * 80)
    print("✅ وضعیت خوب است:")
    print(f"   بلوک‌های ۰ تا {last_common_complete} کامل هستند.")
    print(f"   پردازش از بلوک {last_common_complete + 1} ادامه می‌یابد.")
    print("   (مشاهده لاگ نشان می‌دهد که بلوک ۶۷ در حال پردازش است)")
    print("=" * 80)

ds.close()