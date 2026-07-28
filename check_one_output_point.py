#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

zarr_path = r"I:\climatology_366_rolling\climatology_stationwise_final.zarr"
ds = xr.open_zarr(zarr_path, consolidated=False)

# نقاط مختلف را بررسی کنید
points_to_check = [0, 100, 1000, 5000, 10000]

for point_idx in points_to_check:
    if point_idx >= ds.dims['point']:
        continue
    data = ds.isel(point=point_idx)
    best = data["tmean_best_dist"].values
    non_failed = np.sum(best != -1)
    print(f"Point {point_idx}: {non_failed}/366 days have valid fit (non-failed)")

    # اگر حداقل یک روز موفق بود، یک نمونه نشان بده
    if non_failed > 0:
        # اولین روز موفق را پیدا کن
        day_idx = np.where(best != -1)[0][0]
        print(f"   Example day {day_idx}: best_dist={best[day_idx]}")
        # پارامترهای مربوطه را نشان بده
        if best[day_idx] == 0:
            print(f"      Normal: mean={data['tmean_normal_p1'].values[day_idx]:.2f}, std={data['tmean_normal_p2'].values[day_idx]:.2f}")
        elif best[day_idx] == 1:
            print(f"      Skew-Normal: alpha={data['tmean_skewnormal_p1'].values[day_idx]:.2f}")
        elif best[day_idx] == 2:
            print(f"      Bimodal: mu1={data['tmean_bimodal_p2'].values[day_idx]:.2f}, mu2={data['tmean_bimodal_p4'].values[day_idx]:.2f}")
        elif best[day_idx] == 3:
            print(f"      Pearson: shape={data['tmean_pearson_p1'].values[day_idx]:.2f}")

ds.close()