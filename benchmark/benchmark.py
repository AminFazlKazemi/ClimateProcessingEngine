#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark/benchmark.py
Performance benchmark for distributions
"""

import time
import numpy as np
from plugins.distributions.normal import NormalDistribution
from plugins.distributions.skewnormal import SkewNormalDistribution
from plugins.distributions.gev import GEVDistribution

def benchmark():
    data = np.random.randn(1000)
    dists = [NormalDistribution(), SkewNormalDistribution(), GEVDistribution()]
    for dist in dists:
        t0 = time.time()
        res = dist.fit(data)
        t1 = time.time()
        print(f"{dist.name}: {t1-t0:.4f} sec")

if __name__ == "__main__":
    benchmark()
