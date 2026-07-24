#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
result_pipeline/validate_result.py
================================================================================
اعتبارسنجی نتایج قبل از نوشتن در Zarr.
فقط گزارش می‌دهد.
================================================================================
"""

import numpy as np
from zarr_schema import VAR_NAMES, VAR_DTYPES
from constants import N_DAYS, VALID_BEST_DIST

def validate_result(block_result, block_start, block_size, strict=True):
    report = {"valid": True, "errors": [], "warnings": [], "stats": {}}
    for name in VAR_NAMES:
        arr = block_result[name]
        if arr.shape != (N_DAYS, block_size):
            report["valid"] = False
            report["errors"].append(f"{name}: shape {arr.shape} != ({N_DAYS}, {block_size})")
            continue
        expected_dtype = np.dtype(VAR_DTYPES[name])
        if arr.dtype != expected_dtype:
            report["warnings"].append(f"{name}: dtype {arr.dtype} != {expected_dtype}")
        if name == "best_dist":
            invalid = set(np.unique(arr)) - VALID_BEST_DIST
            if invalid:
                report["valid"] = False
                report["errors"].append(f"best_dist contains invalid values: {invalid}")
        elif name == "count":
            if np.any(arr < 0):
                report["valid"] = False
                report["errors"].append("count contains negative values")
            if np.any(arr > 155):
                report["warnings"].append(f"count > 155 (max={np.max(arr)})")
        elif name == "std":
            if np.any(arr < 0):
                report["valid"] = False
                report["errors"].append("std contains negative values")
            if np.any(np.isinf(arr)):
                report["valid"] = False
                report["errors"].append(f"{name} contains Inf")
        elif "loglik" in name or "aicc" in name or "bic" in name:
            if np.any(np.isinf(arr)):
                report["warnings"].append(f"{name} contains Inf")
        report["stats"][name] = {
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "nan_ratio": float(np.isnan(arr).sum() / arr.size),
        }
    if strict and not report["valid"]:
        raise ValueError("Result validation failed:\n" + "\n".join(report["errors"]))
    return report

def print_validation_report(report, prefix="   "):
    if report["valid"]:
        print(f"{prefix}✅ Result validation passed.")
    else:
        print(f"{prefix}❌ Result validation failed.")
    if report["warnings"]:
        print(f"{prefix}   ⚠️ {len(report['warnings'])} warnings:")
        for w in report["warnings"][:5]:
            print(f"{prefix}      {w}")
    if report["errors"]:
        print(f"{prefix}   ❌ {len(report['errors'])} errors:")
        for e in report["errors"][:5]:
            print(f"{prefix}      {e}")
