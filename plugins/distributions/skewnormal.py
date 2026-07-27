#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/distributions/skewnormal.py
Skew-Normal distribution plugin
"""

import numpy as np
from core.engine.distribution_plugin import DistributionPlugin

class SkewNormalDistribution(DistributionPlugin):
    name = "Skew"
    code = 1
    params = ["p1", "p2", "p3"]
    n_params = 3
    supports_negative = True
    supports_zero = True
    supports_positive = True

    def fit(self, data):
        mean = np.mean(data)
        std = np.std(data)
        skew = np.mean(((data - mean)/std)**3) if std > 0 else 0
        delta = skew / np.sqrt(1 + (2/np.pi - 1)*skew**2) if abs(skew) < 0.99 else 0
        omega = std / np.sqrt(1 - 2*delta**2/np.pi)
        xi = mean - omega * delta * np.sqrt(2/np.pi)
        alpha = delta / np.sqrt(1 - delta**2) if abs(delta) < 0.999 else 0
        loglik = -len(data)*np.log(omega) + np.sum(np.log(np.exp(-0.5*((data-xi)/omega)**2)/np.sqrt(2*np.pi) * 
                      (1 + np.sign(alpha*(data-xi)/omega) * (1 - np.exp(-2*(alpha*(data-xi)/omega)**2/np.pi))**0.5)))
        return {
            "p1": xi,
            "p2": omega,
            "p3": alpha,
            "loglik": loglik,
            "aicc": -2*loglik + 2*self.n_params + (2*self.n_params*(self.n_params+1))/(len(data)-self.n_params-1),
            "bic": -2*loglik + self.n_params*np.log(len(data)),
        }
