#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core/engine/distribution_plugin.py
Base class for distribution plugins
"""

class DistributionPlugin:
    name = None
    code = None
    params = []
    n_params = 0
    supports_negative = False
    supports_zero = False
    supports_positive = False
    extreme_only = False

    def fit(self, data):
        raise NotImplementedError
