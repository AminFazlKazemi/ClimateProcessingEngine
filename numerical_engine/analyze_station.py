#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/analyze_station.py
تحلیل یک ایستگاه با خروجی دیکشنری
"""

import numpy as np
from numerical_engine.window_engine import extract_window_values_fast
from numerical_engine.distributions import fit_distribution
from zarr_schema import VAR_NAMES, N_DAYS

def analyze_station(station_data, window_table, var_idx=0):
    # مقداردهی اولیه با NaN
    result = {}
    for name in VAR_NAMES:
        if name == "best_dist":
            result[name] = np.full(N_DAYS, -1, dtype=np.int32)
        elif name == "count":
            result[name] = np.zeros(N_DAYS, dtype=np.int32)
        else:
            result[name] = np.full(N_DAYS, np.nan, dtype=np.float32)

    windows = extract_window_values_fast(station_data, window_table, var_idx)
    for doy_idx, values in enumerate(windows):
        if values is None:
            continue
        res = fit_distribution(values)
        if res is None:
            continue
        # res یک دیکشنری است
        for key, val in res.items():
            if key in result:
                if key == "count":
                    result[key][doy_idx] = int(val)
                else:
                    result[key][doy_idx] = val
    return result
