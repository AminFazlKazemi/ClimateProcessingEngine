#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check the first point in the output Zarr to verify that fitting succeeded.
Run this script while main.py is running or after it finishes.
"""

import time
import numpy as np
import xarray as xr
from pathlib import Path

zarr_path = Path(r"I:\climatology_366_rolling\climatology_stationwise_final.zarr")

def check_first_point():
    """بررسی نقطه اول و نمایش وضعیت آن"""
    
    if not zarr_path.exists():
        print("⏳ Zarr store not created yet. Waiting...")
        return False
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
    except Exception as e:
        print(f"⏳ Zarr store is being written or not ready: {e}")
        return False
    
    # نقطه اول
    point_idx = 0
    data = ds.isel(point=point_idx)
    
    # بررسی tmean_best_dist
    best = data["tmean_best_dist"].values
    non_failed = np.sum(best != -1)
    total_days = len(best)
    
    print(f"\n📊 Point 0 ({point_idx}):")
    print(f"   Total days: {total_days}")
    print(f"   Days with valid fit (best_dist != -1): {non_failed}")
    print(f"   Percentage: {100 * non_failed / total_days:.1f}%")
    
    if non_failed == 0:
        print("   ❌ No valid fits found for point 0!")
        ds.close()
        return False
    
    # پیدا کردن اولین روز موفق
    first_success = np.where(best != -1)[0][0]
    best_code = best[first_success]
    dist_names = {-1: "Failed", 0: "Normal", 1: "Skew-Normal", 2: "Bimodal", 3: "Pearson III"}
    
    print(f"\n   ✅ First successful day: {first_success}")
    print(f"      Best distribution: {dist_names.get(best_code, 'Unknown')} (code {best_code})")
    
    # بررسی پارامترهای tmean
    mean_vals = data["tmean_mean"].values
    if not np.all(np.isnan(mean_vals)):
        valid_means = mean_vals[~np.isnan(mean_vals)]
        print(f"      tmean_mean: {len(valid_means)} valid days")
        print(f"         min: {np.min(valid_means):.2f} °C, max: {np.max(valid_means):.2f} °C")
        print(f"         mean: {np.mean(valid_means):.2f} °C")
    
    # بررسی پارامترهای توزیع برای روز موفق
    if best_code == 0:  # Normal
        p1 = data["tmean_normal_p1"].values[first_success]
        p2 = data["tmean_normal_p2"].values[first_success]
        print(f"      Normal parameters (day {first_success}):")
        print(f"         mean (μ): {p1:.2f}")
        print(f"         std (σ): {p2:.2f}")
    elif best_code == 1:  # Skew-Normal
        p1 = data["tmean_skewnormal_p1"].values[first_success]
        p2 = data["tmean_skewnormal_p2"].values[first_success]
        p3 = data["tmean_skewnormal_p3"].values[first_success]
        print(f"      Skew-Normal parameters (day {first_success}):")
        print(f"         alpha: {p1:.2f}")
        print(f"         loc: {p2:.2f}")
        print(f"         scale: {p3:.2f}")
    elif best_code == 2:  # Bimodal
        p1 = data["tmean_bimodal_p1"].values[first_success]
        p2 = data["tmean_bimodal_p2"].values[first_success]
        p3 = data["tmean_bimodal_p3"].values[first_success]
        p4 = data["tmean_bimodal_p4"].values[first_success]
        p5 = data["tmean_bimodal_p5"].values[first_success]
        print(f"      Bimodal parameters (day {first_success}):")
        print(f"         w1: {p1:.2f}")
        print(f"         mu1: {p2:.2f}")
        print(f"         sigma1: {p3:.2f}")
        print(f"         mu2: {p4:.2f}")
        print(f"         sigma2: {p5:.2f}")
    elif best_code == 3:  # Pearson III
        p1 = data["tmean_pearson_p1"].values[first_success]
        p2 = data["tmean_pearson_p2"].values[first_success]
        p3 = data["tmean_pearson_p3"].values[first_success]
        print(f"      Pearson III parameters (day {first_success}):")
        print(f"         shape: {p1:.2f}")
        print(f"         scale: {p2:.2f}")
        print(f"         loc: {p3:.2f}")
    
    print("\n✅ Point 0 has valid data. Processing is working correctly!")
    ds.close()
    return True

# اجرای چک
if __name__ == "__main__":
    # اگر Zarr store وجود ندارد، چند بار تلاش کن
    attempts = 0
    max_attempts = 30
    while attempts < max_attempts:
        if check_first_point():
            break
        attempts += 1
        time.sleep(10)  # هر ۱۰ ثانیه یک بار بررسی کن
    else:
        print("\n⏳ Zarr store not ready after 5 minutes. Check later.")