#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
result_pipeline/write_block.py
================================================================================
نوشتن نتایج یک بلوک در Zarr – نسخه‌ی همزمان (بدون Async)
برای جلوگیری از خطای ThreadPoolExecutor shutdown
================================================================================
"""

from zarr_schema import VAR_NAMES


def write_block(root, block_result, block_start, block_end):
    """
    نوشتن همزمان داده‌ها در Zarr.
    
    پارامترها:
        root: گروه Zarr (zarr.Group)
        block_result: دیکشنری شامل آرایه‌های (day, block_size) برای هر متغیر
        block_start: اندیس شروع در محور point
        block_end: اندیس پایان (اختصاصی)
    
    بازگشت:
        None
    """
    for name in VAR_NAMES:
        root[name][:, block_start:block_end] = block_result[name]


def write_block_safe(root, block_result, block_start, block_end, validate=True, async_mode=False):
    """
    نوشتن با اعتبارسنجی (اختیاری) – فقط همزمان.
    پارامتر async_mode برای سازگاری نگه‌داشته شده اما همیشه False است.
    
    پارامترها:
        root: گروه Zarr
        block_result: دیکشنری نتایج بلوک
        block_start: اندیس شروع
        block_end: اندیس پایان
        validate: اگر True باشد، قبل از نوشتن اعتبارسنجی انجام می‌شود
        async_mode: برای سازگاری نگه‌داشته شده (همیشه False)
    
    بازگشت:
        True در صورت موفقیت
    
    استثناها:
        ValueError: اگر اعتبارسنجی شکست بخورد
    """
    if validate:
        from result_pipeline.validate_result import validate_result
        report = validate_result(block_result, block_start, block_end - block_start)
        if not report["valid"]:
            raise ValueError("Cannot write: validation failed")

    # نوشتن همزمان
    write_block(root, block_result, block_start, block_end)
    return True