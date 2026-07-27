#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/quality/quality_flag.py
Quality flag system for fits
"""

import numpy as np

class QualityFlag:
    PASS = 0
    LOW_SAMPLE = 1
    NO_CONVERGENCE = 2
    OUTLIER = 3
    HIGH_AICC = 4
    BAD_SKEW = 5
    NAN_INPUT = 6
    INF_INPUT = 7

    @staticmethod
    def evaluate(fit_result, data, threshold_aicc=1000):
        flags = []
        if len(data) < 3:
            flags.append(QualityFlag.LOW_SAMPLE)
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            flags.append(QualityFlag.NAN_INPUT)
        if fit_result.get("aicc", np.inf) > threshold_aicc:
            flags.append(QualityFlag.HIGH_AICC)
        return flags if flags else [QualityFlag.PASS]
