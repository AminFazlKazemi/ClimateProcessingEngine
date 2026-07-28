#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
result_pipeline/write_block.py
================================================================================
نوشتن نتایج یک بلوک در Zarr.
فقط ۳۳ write برای هر بلوک. هیچ محاسبه‌ای انجام نمی‌دهد.
================================================================================
"""

from zarr_schema import VAR_NAMES

def write_block(root, block_result, block_start, block_end):
    for name in VAR_NAMES:
        root[name][:, block_start:block_end] = block_result[name]

def write_block_safe(root, block_result, block_start, block_end, validate=True):
    if validate:
        from result_pipeline.validate_result import validate_result
        report = validate_result(block_result, block_start, block_end - block_start)
        if not report["valid"]:
            raise ValueError("Cannot write: validation failed")
    write_block(root, block_result, block_start, block_end)