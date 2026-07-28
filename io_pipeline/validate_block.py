# -*- coding: utf-8 -*-
"""
validate_block.py - اعتبارسنجی داده‌های بلوک
"""

import numpy as np
from constants import N_DAYS, N_VARS, FLOAT_DTYPE

def validate_block(block_data, block_start, block_size, strict=True):
    """
    اعتبارسنجی داده‌های یک بلوک
    
    Parameters:
        block_data: ndarray (block_size, n_years, N_DAYS, N_VARS)
        block_start: int
        block_size: int
        strict: bool – اگر True باشد، خطا raise می‌کند
    
    Returns:
        dict: report with valid, errors, warnings, stats
    """
    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    # 1. بررسی نوع داده
    if block_data.dtype != FLOAT_DTYPE:
        report["errors"].append(f"dtype: {block_data.dtype} != {FLOAT_DTYPE}")
        report["valid"] = False
    
    # 2. بررسی ابعاد (انعطاف‌پذیر)
    if block_data.ndim != 4:
        report["errors"].append(f"ndim: {block_data.ndim} != 4")
        report["valid"] = False
    else:
        bsize, n_years, n_days, n_vars = block_data.shape
        report["stats"]["shape"] = block_data.shape
        
        if bsize != block_size:
            report["errors"].append(f"block_size: {bsize} != {block_size}")
            report["valid"] = False
        
        if n_days != N_DAYS:
            report["errors"].append(f"n_days: {n_days} != {N_DAYS}")
            report["valid"] = False
        
        if n_vars != N_VARS:
            report["errors"].append(f"n_vars: {n_vars} != {N_VARS}")
            report["valid"] = False
        
        # n_years می‌تواند هر عددی باشد، فقط آن را ثبت می‌کنیم
        report["stats"]["n_years"] = n_years
    
    # 3. بررسی contiguous
    if not block_data.flags['C_CONTIGUOUS']:
        report["warnings"].append("block_data is not C-contiguous")
        if strict:
            report["errors"].append("block_data is not C-contiguous")
            report["valid"] = False
    
    # 4. بررسی NaN
    total = block_data.size
    nan_count = np.count_nonzero(np.isnan(block_data))
    nan_ratio = nan_count / total if total > 0 else 1.0
    report["stats"]["nan_ratio"] = nan_ratio
    report["stats"]["nan_count"] = nan_count
    
    if nan_ratio > 0.99:
        report["warnings"].append(f"NaN ratio: {nan_ratio:.2%}")
    
    # 5. بررسی محدوده (اگر داده وجود داشته باشد)
    finite_data = block_data[~np.isnan(block_data)]
    if len(finite_data) > 0:
        report["stats"]["min_val"] = float(np.min(finite_data))
        report["stats"]["max_val"] = float(np.max(finite_data))
        report["stats"]["mean_val"] = float(np.mean(finite_data))
    else:
        report["warnings"].append("All data is NaN")
    
    # 6. چاپ گزارش
    print("   📊 Validation Report:")
    if report["valid"]:
        print(f"      shape: {block_data.shape}")
        print(f"      dtype: {block_data.dtype}")
        print(f"      contiguous: {block_data.flags['C_CONTIGUOUS']}")
        print(f"      NaN ratio: {nan_ratio:.2%}")
        if len(finite_data) > 0:
            print(f"      range: [{report['stats']['min_val']:.2f}, {report['stats']['max_val']:.2f}]")
    if report["warnings"]:
        print(f"      ⚠️ warnings: {len(report['warnings'])}")
    if report["errors"]:
        print(f"      ❌ errors: {len(report['errors'])}")
        for err in report["errors"]:
            print(f"         {err}")
    
    if not report["valid"] and strict:
        raise ValueError("Block validation failed:\n" + "\n".join(report["errors"]))
    
    return report

def print_validation_report(report):
    """چاپ زیبای گزارش اعتبارسنجی"""
    print("   📊 Validation Report:")
    if report.get("stats", {}).get("shape"):
        print(f"      shape: {report['stats']['shape']}")
    if report.get("stats", {}).get("nan_ratio") is not None:
        print(f"      NaN ratio: {report['stats']['nan_ratio']:.2%}")
    if report.get("errors"):
        print(f"      ❌ errors: {len(report['errors'])}")
    if report.get("warnings"):
        print(f"      ⚠️ warnings: {len(report['warnings'])}")