#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/distributions/normal.py
Normal distribution plugin
"""

import numpy as np
from core.engine.distribution_plugin import DistributionPlugin

class NormalDistribution(DistributionPlugin):
    name = "Normal"
    code = 0
    params = ["p1", "p2"]
    n_params = 2
    supports_negative = True
    supports_zero = True
    supports_positive = True

    def fit(self, data):
        mean = np.mean(data)
        std = np.std(data)
        loglik = -0.5 * len(data) * np.log(2*np.pi*std**2) - np.sum((data-mean)**2)/(2*std**2)
        return {
            "p1": mean,
            "p2": std,
            "loglik": loglik,
            "aicc": -2*loglik + 2*self.n_params + (2*self.n_params*(self.n_params+1))/(len(data)-self.n_params-1),
            "bic": -2*loglik + self.n_params*np.log(len(data)),
        }
