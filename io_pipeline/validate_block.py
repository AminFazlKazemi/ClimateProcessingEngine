#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
io_pipeline/validate_block.py
================================================================================
اعتبارسنجی داده‌های بارگذاری‌شده (Data Contract).
فقط گزارش می‌دهد، اصلاح نمی‌کند.
================================================================================
ورژن: 2.0 - نهایی
"""

import numpy as np
from constants import N_YEARS, N_DAYS, N_VARS, FLOAT_DTYPE

def validate_block(block_data, block_start, block_size, strict=True):
    report = {
        "shape": block_data.shape,
        "dtype": block_data.dtype,
        "contiguous": block_data.flags["C_CONTIGUOUS"],
        "nan_ratio": float(np.isnan(block_data).sum() / block_data.size),
        "min_val": float(np.nanmin(block_data)),
        "max_val": float(np.nanmax(block_data)),
        "finite": np.all(np.isfinite(block_data[~np.isnan(block_data)])),
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    expected_shape = (block_size, N_YEARS, N_DAYS, N_VARS)
    if block_data.shape != expected_shape:
        report["valid"] = False
        report["errors"].append(f"shape: {block_data.shape} != {expected_shape}")
    if block_data.dtype != FLOAT_DTYPE:
        report["warnings"].append(f"dtype: {block_data.dtype} != {FLOAT_DTYPE}")
    if not block_data.flags["C_CONTIGUOUS"]:
        report["warnings"].append("block_data is not C-contiguous")
    if report["nan_ratio"] > 0.5:
        report["warnings"].append(f"NaN ratio > 50%: {report['nan_ratio']:.2%}")
    if not report["finite"]:
        report["valid"] = False
        report["errors"].append("contains Inf or -Inf")
    for i in range(min(block_size, 10)):
        station_data = block_data[i]
        valid_days = (~np.isnan(station_data)).all(axis=(0, 2))
        valid_years = (~np.isnan(station_data)).all(axis=(1, 2))
        if valid_days.sum() < N_DAYS * 0.5:
            report["warnings"].append(f"station {i}: only {valid_days.sum()}/{N_DAYS} days valid")
        if valid_years.sum() < N_YEARS * 0.5:
            report["warnings"].append(f"station {i}: only {valid_years.sum()}/{N_YEARS} years valid")
    if strict and not report["valid"]:
        raise ValueError("Block validation failed:\n" + "\n".join(report["errors"]))
    return report

def print_validation_report(report):
    print(f"   📊 Validation Report:")
    print(f"      shape: {report['shape']}")
    print(f"      dtype: {report['dtype']}")
    print(f"      contiguous: {report['contiguous']}")
    print(f"      NaN ratio: {report['nan_ratio']:.2%}")
    print(f"      range: [{report['min_val']:.1f}, {report['max_val']:.1f}]")
    if report["warnings"]:
        print(f"      ⚠️ warnings: {len(report['warnings'])}")
    if report["errors"]:
        print(f"      ❌ errors: {len(report['errors'])}")
