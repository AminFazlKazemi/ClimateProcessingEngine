#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monitoring/benchmark.py
================================================================================
Benchmark خودکار با RAM monitoring و Failed Fits.
================================================================================
"""

import time
from constants import BENCHMARK_TEST_SIZES

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil نصب نیست. RAM monitoring غیرفعال است.")

def run_benchmark(load_func, analyze_func, write_func, test_sizes=None):
    if test_sizes is None:
        test_sizes = BENCHMARK_TEST_SIZES
    results = {}
    for size in test_sizes:
        print(f"   📊 Testing block size: {size}")
        mem_before = psutil.virtual_memory().used / (1024**3) if HAS_PSUTIL else 0
        t0 = time.time()
        block_data = load_func(0, size)
        t_load = time.time() - t0
        t0 = time.time()
        block_result = analyze_func(block_data)
        t_analyze = time.time() - t0
        t0 = time.time()
        write_func(block_result)
        t_write = time.time() - t0
        mem_after = psutil.virtual_memory().used / (1024**3) if HAS_PSUTIL else 0
        failed_fits = int((block_result["best_dist"] == -1).sum())
        total_fits = size * 366
        results[size] = {
            "load": t_load,
            "analyze": t_analyze,
            "write": t_write,
            "total": t_load + t_analyze + t_write,
            "stations_per_sec": size / t_analyze if t_analyze > 0 else 0,
            "ram_peak_gb": max(mem_before, mem_after),
            "failed_fits": failed_fits,
            "failed_ratio": (failed_fits / total_fits * 100) if total_fits > 0 else 0,
        }
        print(f"      Load: {t_load:.1f}s | Analyze: {t_analyze:.1f}s | Write: {t_write:.1f}s")
        print(f"      Stations/sec: {results[size]['stations_per_sec']:.1f}")
        print(f"      RAM peak: {results[size]['ram_peak_gb']:.2f} GB | Failed: {failed_fits} ({results[size]['failed_ratio']:.2f}%)")
    best_size = min(results.keys(), key=lambda s: results[s]["total"])
    return {"results": results, "recommended": best_size, "best_stats": results[best_size]}

def print_benchmark_report(report):
    print("\n" + "="*70)
    print("📊 BENCHMARK REPORT")
    print("="*70)
    for size, stats in report["results"].items():
        print(f"   Size {size:4d}: Load {stats['load']:6.1f}s | Analyze {stats['analyze']:6.1f}s | Write {stats['write']:5.1f}s | {stats['stations_per_sec']:5.1f} st/s | RAM {stats['ram_peak_gb']:.2f}GB | Failed {stats['failed_fits']:5d}")
    print("-"*70)
    print(f"✅ Recommended: {report['recommended']}")
    print(f"   Best speed: {report['best_stats']['stations_per_sec']:.1f} stations/sec")
    print(f"   RAM usage: {report['best_stats']['ram_peak_gb']:.2f} GB")
    print("="*70)
