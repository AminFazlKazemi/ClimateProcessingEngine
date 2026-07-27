#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/distributions/gev.py
GEV distribution plugin (extreme values only)
"""

import numpy as np
from core.engine.distribution_plugin import DistributionPlugin

class GEVDistribution(DistributionPlugin):
    name = "GEV"
    code = 4
    params = ["p1", "p2", "p3"]
    n_params = 3
    supports_negative = True
    supports_zero = True
    supports_positive = True
    extreme_only = True

    def fit(self, data):
        mean = np.mean(data)
        std = np.std(data)
        skew = np.mean(((data - mean)/std)**3) if std > 0 else 0
        k = 0.5 * (1 - (3*skew - 1)/(3*skew + 1)) if skew != 0 else 0
        k = np.clip(k, -0.5, 0.5)
        scale = std * k / (np.sqrt(np.exp(np.log(1+2*k)-2*np.log(1+k))) if k != 0 else std)
        scale = np.clip(scale, 1e-6, 1e6)
        loc = mean - scale * (np.exp(np.log(1+k)) - 1) / k if k != 0 else mean
        z = (data - loc) / scale
        if k != 0:
            z = 1 - k*z
            if np.any(z <= 0):
                loglik = -1e100
            else:
                loglik = -len(data)*np.log(scale) - (1-1/k)*np.sum(np.log(z)) - np.sum(np.exp(-1/k*np.log(z)))
        else:
            loglik = -len(data)*np.log(scale) - np.sum(z) - np.sum(np.exp(-z))
        return {
            "p1": loc,
            "p2": scale,
            "p3": k,
            "loglik": loglik,
            "aicc": -2*loglik + 2*self.n_params + (2*self.n_params*(self.n_params+1))/(len(data)-self.n_params-1),
            "bic": -2*loglik + self.n_params*np.log(len(data)),
        }
