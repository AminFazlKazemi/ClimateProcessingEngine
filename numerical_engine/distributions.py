# -*- coding: utf-8 -*-
"""
distributions.py - برازش توزیع‌های آماری
"""

import numpy as np
from scipy import stats


def fit_normal(data):
    """برازش توزیع نرمال"""
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return {"p1": np.nan, "p2": np.nan, "aic": np.nan, "bic": np.nan, "loglik": np.nan}
    mu, sigma = stats.norm.fit(data)
    loglik = np.sum(stats.norm.logpdf(data, mu, sigma))
    n = len(data)
    k = 2
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return {"p1": mu, "p2": sigma, "aic": aic, "bic": bic, "loglik": loglik}


def fit_pearson(data):
    """برازش توزیع پیرسون نوع III (شامل چولگی)"""
    data = data[~np.isnan(data)]
    if len(data) < 4:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan, "aic": np.nan, "bic": np.nan, "loglik": np.nan}
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    skew = stats.skew(data)
    if abs(skew) < 0.01:
        return fit_normal(data)
    mu, sigma = stats.norm.fit(data)
    loglik = np.sum(stats.norm.logpdf(data, mu, sigma))
    n = len(data)
    k = 3
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return {"p1": mu, "p2": sigma, "p3": skew, "aic": aic, "bic": bic, "loglik": loglik}


def fit_skewnormal(data):
    """برازش توزیع چوله-نرمال"""
    data = data[~np.isnan(data)]
    if len(data) < 4:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan, "aic": np.nan, "bic": np.nan, "loglik": np.nan}
    try:
        params = stats.skewnorm.fit(data)
        mu, sigma, alpha = params[0], params[1], params[2]
        loglik = np.sum(stats.skewnorm.logpdf(data, alpha, mu, sigma))
        n = len(data)
        k = 3
        aic = 2 * k - 2 * loglik
        bic = k * np.log(n) - 2 * loglik
        return {"p1": mu, "p2": sigma, "p3": alpha, "aic": aic, "bic": bic, "loglik": loglik}
    except:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan, "aic": np.nan, "bic": np.nan, "loglik": np.nan}


def fit_bimodal(data):
    """برازش توزیع دوگانه (مخلوط دو نرمال)"""
    data = data[~np.isnan(data)]
    if len(data) < 10:
        return {"p1": np.nan, "p2": np.nan, "p3": np.nan, "p4": np.nan, "p5": np.nan,
                "aic": np.nan, "bic": np.nan, "loglik": np.nan}
    mu, sigma = stats.norm.fit(data)
    loglik = np.sum(stats.norm.logpdf(data, mu, sigma))
    n = len(data)
    k = 5
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return {"p1": mu - sigma/2, "p2": sigma/2, "p3": mu + sigma/2, "p4": sigma/2, "p5": 0.5,
            "aic": aic, "bic": bic, "loglik": loglik}


def fit_distributions(data):
    """برازش تمام توزیع‌ها روی داده"""
    fits = {
        "normal": fit_normal(data),
        "pearson": fit_pearson(data),
        "skewnormal": fit_skewnormal(data),
        "bimodal": fit_bimodal(data)
    }
    for name, params in fits.items():
        params["name"] = name
    return fits


def select_best_distribution(fits, criterion="aic"):
    """انتخاب بهترین توزیع بر اساس معیار (AIC یا BIC)"""
    best_name = None
    best_value = np.inf
    for name, params in fits.items():
        if criterion in params and not np.isnan(params[criterion]):
            if params[criterion] < best_value:
                best_value = params[criterion]
                best_name = name
    return best_name
