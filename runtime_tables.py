#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_tables.py
================================================================================
ساخت جداول زمان اجرا (مستقل از تقویم).
شامل: window_table, file_map, variable_table
================================================================================
ورژن: 2.0 - نهایی
"""

import os
import glob
import numpy as np
from constants import N_DAYS, WINDOW_DAYS, ZARR_BASE, VARS

def build_window_table():
    window_table = []
    for doy in range(N_DAYS):
        indices = []
        for d in range(doy - WINDOW_DAYS, doy + WINDOW_DAYS + 1):
            if d < 0:
                d_adj = N_DAYS + d
            elif d >= N_DAYS:
                d_adj = d - N_DAYS
            else:
                d_adj = d
            indices.append(d_adj)
        window_table.append(indices)
    return window_table

def build_file_map(zarr_base=None):
    if zarr_base is None:
        zarr_base = ZARR_BASE
    file_map = {}
    zarr_files = glob.glob(os.path.join(zarr_base, "*.zarr"))
    for f in zarr_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) >= 3:
            try:
                year = int(parts[0])
                month = int(parts[1])
                file_map[(year, month)] = f
            except ValueError:
                continue
    return file_map

def build_variable_table():
    var_index = {name: idx for idx, name in enumerate(VARS)}
    return {"var_index": var_index, "var_names": VARS, "n_vars": len(VARS)}

def build_runtime_tables(zarr_base=None):
    return {
        "window_table": build_window_table(),
        "file_map": build_file_map(zarr_base),
        "variable_table": build_variable_table(),
    }

def print_runtime_info(tables):
    print(f"   window_table: {len(tables['window_table'])} روز")
    print(f"   file_map: {len(tables['file_map'])} فایل")
    print(f"   variables: {tables['variable_table']['n_vars']} متغیر")

if __name__ == "__main__":
    tables = build_runtime_tables()
    print("✅ Runtime Tables:")
    print_runtime_info(tables)
