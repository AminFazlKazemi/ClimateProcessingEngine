#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/uncertainty/bootstrap.py
Bootstrap uncertainty estimation
"""

import numpy as np

def bootstrap_fit(data, fit_func, n_bootstrap=100, confidence=0.95):
    n = len(data)
    results = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        res = fit_func(sample)
        results.append(res)
    param_names = list(results[0].keys())
    cis = {}
    for p in param_names:
        vals = [r[p] for r in results if not np.isnan(r[p])]
        if vals:
            lower = np.percentile(vals, (1-confidence)/2 * 100)
            upper = np.percentile(vals, (1+confidence)/2 * 100)
            cis[p] = {"mean": np.mean(vals), "lower": lower, "upper": upper}
    return cis
