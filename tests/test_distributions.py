#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_distributions.py
Unit tests for distributions
"""

import pytest
import numpy as np
from plugins.distributions.normal import NormalDistribution

def test_normal_fit():
    data = np.random.normal(10, 2, 100)
    dist = NormalDistribution()
    res = dist.fit(data)
    assert abs(res["p1"] - 10) < 0.5
    assert abs(res["p2"] - 2) < 0.5
