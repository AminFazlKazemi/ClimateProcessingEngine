#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/distributions.py
================================================================================
توابع برازش ۴ توزیع آماری با Numba.
کاملاً مستقل از هر چیز دیگر. فقط NumPy و Numba.
================================================================================
ورژن: 2.1 - رفع خطای Numba
"""

import numpy as np
from numba import njit
import math
from constants import MIN_VALID_VALUES

# ============================================================================
# توابع کمکی
# ============================================================================
@njit
def compute_stats_numba(data):
    n = len(data)
    if n < MIN_VALID_VALUES:
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = np.mean(data)
    std = np.std(data)
    sorted_data = np.sort(data)
    median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    skew = np.mean(((data - mean) / std) ** 3) if std > 0 else 0.0
    return mean, std, skew, median, n

@njit
def compute_aicc_numba(loglik, k, n):
    aic = 2 * k - 2 * loglik
    if n - k - 1 <= 0:
        return aic
    return aic + (2 * k * (k + 1)) / (n - k - 1)

@njit
def compute_bic_numba(loglik, k, n):
    return k * np.log(n) - 2 * loglik

# ============================================================================
# توزیع نرمال
# ============================================================================
@njit
def logpdf_normal_numba(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2

@njit
def fit_normal_full_numba(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 2
    loglik = 0.0
    for i in range(n):
        loglik += logpdf_normal_numba(data[i], mean, std)
    aicc = compute_aicc_numba(loglik, 2, n)
    bic = compute_bic_numba(loglik, 2, n)
    return mean, std, loglik, aicc, bic, 2

# ============================================================================
# توزیع Skew-normal
# ============================================================================
@njit
def skewness_to_alpha(skew):
    if abs(skew) < 0.1:
        return 0.0
    elif abs(skew) < 0.3:
        return 0.5 * skew
    elif abs(skew) < 0.6:
        return 1.0 * skew
    else:
        return 2.0 * skew

@njit
def skewnorm_logpdf_scalar(x, a, loc, scale):
    z = (x - loc) / scale
    phi = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
    Phi = 0.5 * (1 + math.erf(a * z / math.sqrt(2)))
    if Phi <= 0:
        return -np.inf
    return math.log(2) + math.log(phi) + math.log(Phi) - math.log(scale)

@njit
def fit_skewnorm_full_numba(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    skew = np.mean(((data - mean) / std) ** 3)
    if abs(skew) >= 0.99:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    alpha = skewness_to_alpha(skew)
    alpha = max(-10.0, min(10.0, alpha))
    delta = alpha / np.sqrt(1 + alpha**2)
    scale = std / np.sqrt(1 - 2 * delta**2 / np.pi)
    loc = mean - scale * delta * np.sqrt(2 / np.pi)
    if scale <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    loglik = 0.0
    for i in range(n):
        val = skewnorm_logpdf_scalar(data[i], alpha, loc, scale)
        if np.isinf(val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
        loglik += val
    aicc = compute_aicc_numba(loglik, 3, n)
    bic = compute_bic_numba(loglik, 3, n)
    return alpha, loc, scale, loglik, aicc, bic, 3

# ============================================================================
# توزیع GEV (کامل - با بهینه‌سازی)
# ============================================================================
@njit
def gev_skewness_func(kisi):
    if abs(kisi) < 1e-8:
        return 1.139547
    g1 = math.gamma(1.0 - kisi)
    g2 = math.gamma(1.0 - 2.0 * kisi)
    g3 = math.gamma(1.0 - 3.0 * kisi)
    if g2 - g1**2 <= 0:
        return np.nan
    skew = (g3 - 3 * g2 * g1 + 2 * g1**3) / ((g2 - g1**2) ** 1.5)
    if kisi < 0:
        skew = -skew
    return skew

@njit
def gev_kisi_optimization(kisi_initial, target_skew, max_iter=20, tol=1e-6):
    kisi = kisi_initial
    if kisi > 0.49:
        kisi = 0.49
    elif kisi < -0.49:
        kisi = -0.49
    for _ in range(max_iter):
        skew_current = gev_skewness_func(kisi)
        if np.isnan(skew_current):
            break
        error = skew_current - target_skew
        if abs(error) < tol:
            break
        delta = 1e-6
        skew_plus = gev_skewness_func(kisi + delta)
        skew_minus = gev_skewness_func(kisi - delta)
        if np.isnan(skew_plus) or np.isnan(skew_minus):
            break
        deriv = (skew_plus - skew_minus) / (2 * delta)
        if abs(deriv) < 1e-12:
            break
        kisi_new = kisi - error / deriv
        if kisi_new > 0.49:
            kisi_new = 0.49
        elif kisi_new < -0.49:
            kisi_new = -0.49
        if abs(kisi_new - kisi) < 1e-8:
            kisi = kisi_new
            break
        kisi = kisi_new
    return kisi

@njit
def gev_initial_kisi(skew):
    if abs(skew) < 1e-6:
        return 0.0
    sign = 1.0 if skew >= 0 else -1.0
    abs_skew = min(abs(skew), 4000.0)
    if abs_skew > 4000:
        x = math.log(6.0 + math.log(1.14 + abs_skew))
        y = 3.378*(x**4) - 34.779*(x**3) + 134.48*(x**2) - 231.98*x + 148.98
        kisi = 1.0 / (3.0 - math.exp(math.exp(math.exp(y))))
    elif abs_skew < 0.867:
        x = 6 + math.log(1.15 - abs_skew)
        if x <= 0.57:
            y = -10.417*(x**2) + 7.2712*x - 0.9757
        else:
            y = 381.97e-4*(x**4) - 887.15e-3*(x**3) + 756.32e-2*(x**2) - 285.19e-1*x + 40.589
        kisi = 1.0 / (3.0 - math.exp(math.exp(math.exp(y))))
    elif abs_skew < 1.134547:
        x = 6 + math.log(1.15 - abs_skew)
        y = 2489.3e-6*(x**6) - 5183.8e-5*(x**5) + 4429.4e-4*(x**4) - 1995.3e-3*(x**3) + 499.84e-2*(x**2) - 67.465e-1*x + 4.5107
        kisi = 1.0 / (3.0 - math.exp(math.exp(math.exp(y))))
    elif abs_skew < 1.21:
        x = math.log(math.log(abs_skew))
        y = 5304.1*(x**6) + 57885*(x**5) + 263034*(x**4) + 637021*(x**3) + 867183*(x**2) + 629153*x + 190056
        kisi = math.exp(-math.exp(y))
    else:
        x = math.log(math.log(abs_skew))
        y = -1767.5e-6*(x**6) - 1201.8e-5*(x**5) + 202.3e-4*(x**4) + 76.848e-3*(x**3) + 3.1281e-2*(x**2) - 5.3997e-1*x + 0.599
        kisi = math.exp(-math.exp(y))
    kisi = sign * kisi
    if kisi < -0.5:
        kisi = -0.5
    elif kisi > 0.5:
        kisi = 0.5
    return kisi

@njit
def gev_logpdf_scalar(x, kisi, loc, scale):
    if scale <= 0:
        return -np.inf
    if abs(kisi) < 1e-8:
        z = (x - loc) / scale
        return -np.log(scale) - z - np.exp(-z)
    z = 1 + kisi * (x - loc) / scale
    if z <= 0:
        return -np.inf
    return -np.log(scale) - (1/kisi + 1) * np.log(z) - z**(-1/kisi)

@njit
def fit_gev_full_numba(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    skew = np.mean(((data - mean) / std) ** 3)
    if skew > 2.999:
        skew = 2.999
    elif skew < -2.999:
        skew = -2.999
    kisi_initial = gev_initial_kisi(skew)
    kisi = gev_kisi_optimization(kisi_initial, skew)
    if abs(kisi) < 1e-4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    g1 = math.gamma(1 - kisi)
    g2 = math.gamma(1 - 2*kisi)
    if g2 - g1*g1 <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    scale = abs(kisi) * std / math.sqrt(g2 - g1*g1)
    loc = mean + scale * (g1 - 1) / kisi
    if scale <= 0 or not math.isfinite(loc):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    loglik = 0.0
    for i in range(n):
        val = gev_logpdf_scalar(data[i], kisi, loc, scale)
        if np.isinf(val):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
        loglik += val
    aicc = compute_aicc_numba(loglik, 3, n)
    bic = compute_bic_numba(loglik, 3, n)
    return kisi, loc, scale, loglik, aicc, bic, 3

# ============================================================================
# توزیع Pearson III
# ============================================================================
@njit
def _norm_ppf_approx(p):
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        a = np.array([2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637], dtype=np.float64)
        b = np.array([-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833], dtype=np.float64)
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
    else:
        r = p if y > 0 else 1-p
        s = math.log(-math.log(r))
        c = np.array([0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                      0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
                      0.0000321767881768, 0.0000002888167364, 0.0000003960315187], dtype=np.float64)
        x = c[0] + s*(c[1] + s*(c[2] + s*(c[3] + s*(c[4] + s*(c[5] + s*(c[6] + s*(c[7] + s*c[8])))))))
        if y < 0:
            x = -x
    return x

@njit
def _gamma_ppf_approx(p, alpha, scale, max_iter=50, tol=1e-7):
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1e10
    z = _norm_ppf_approx(p)
    x0 = alpha * (1 - 1/(9*alpha) + z * math.sqrt(1/(9*alpha)))**3
    x = max(x0, 0.1)
    for _ in range(max_iter):
        logpdf = (alpha-1)*math.log(x) - x - math.lgamma(alpha)
        res = 0.0
        term = 1.0 / alpha
        for i in range(1, 50):
            res += term
            term *= x / (alpha + i)
            if term < 1e-15:
                break
        cdf = 1.0 - math.exp(-x) * (x**alpha) * res / math.exp(math.lgamma(alpha))
        cdf = min(max(cdf, 0.0), 1.0)
        diff = cdf - p
        if abs(diff) < tol:
            break
        pdf = math.exp(logpdf)
        if pdf < 1e-20:
            break
        x = x - diff / pdf
        if x <= 0:
            x = 0.1
    return x * scale

@njit
def _rankdata_max(data):
    n = len(data)
    order = np.argsort(data)
    ranks = np.zeros(n, dtype=np.int32)
    i = 0
    while i < n:
        j = order[i]
        k = i
        while k < n-1 and data[order[k+1]] == data[j]:
            k += 1
        rank_val = k + 1
        for l in range(i, k+1):
            ranks[order[l]] = rank_val
        i = k + 1
    return ranks.astype(np.float64)

@njit
def fit_pearson3_full_numba(data):
    n = len(data)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3

    MEAN_ = np.mean(data)
    STD_ = np.std(data)
    if STD_ == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3

    sumy1 = np.sum(data)
    sumy2 = np.sum(data**2)
    sumy3 = np.sum(data**3)
    SKEW_ = (sumy3 - 3*MEAN_*sumy2 + 3*MEAN_**2*sumy1 - MEAN_**3*n) * n / (n-1) / (n-2) / (STD_**3)

    if SKEW_ > 2.999:
        SKEW = 2.999
    elif SKEW_ < -2.999:
        SKEW = -2.999
    else:
        SKEW = SKEW_

    INVERTED = False
    if SKEW_ < 0:
        DATA_111 = -data.copy()
        INVERTED = True
    else:
        DATA_111 = data.copy()

    minim = np.min(DATA_111)
    DATA_111 = DATA_111 - minim
    DATA_111 = np.where(DATA_111 == 0.0, 0.1, DATA_111)

    XBAR = np.mean(DATA_111)
    STD = np.std(DATA_111)
    lnX_BAR = np.mean(np.log(DATA_111))
    A = np.log(XBAR) - lnX_BAR
    if A <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3

    shape1 = 1/(4*A) * (1 + np.sqrt(1 + 4*A/3))
    scale1 = XBAR / shape1

    q = np.sum(DATA_111 == 0.1) / (n + 0.3*abs(SKEW) + 0.05)
    beg = _gamma_ppf_approx(q, shape1, scale1)
    DATA_111 += beg

    XBAR = np.mean(DATA_111)
    STD = np.std(DATA_111)
    lnX_BAR = np.mean(np.log(DATA_111))
    A = np.log(XBAR) - lnX_BAR
    if A <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3

    shape = 1/(4*A) * (1 + np.sqrt(1 + 4*A/3))
    scale = XBAR / shape

    ranks = _rankdata_max(DATA_111)
    q_ = (ranks - 0.42) / (n + 0.3*SKEW + 0.05)

    CDF = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x = DATA_111[i]
        if x > 0:
            res = 0.0
            term = 1.0 / shape
            for j in range(1, 50):
                res += term
                term *= x / (shape + j)
                if term < 1e-15:
                    break
            cdf_val = 1.0 - math.exp(-x) * (x**shape) * res / math.exp(math.lgamma(shape))
            CDF[i] = min(max(cdf_val, 0.0), 1.0)
        else:
            CDF[i] = 0.0

    if INVERTED:
        CDF = 1.0 - CDF

    loglik = 0.0
    for i in range(n):
        x = DATA_111[i]
        if x <= 0:
            loglik = np.nan
            break
        loglik += (shape - 1)*math.log(x) - x/scale - shape*math.log(scale) - math.lgamma(shape)

    if np.isnan(loglik):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3

    aicc = compute_aicc_numba(loglik, 3, n)
    bic = compute_bic_numba(loglik, 3, n)

    if INVERTED:
        loc_main = -minim
    else:
        loc_main = minim

    return shape, scale, loc_main, loglik, aicc, bic, 3

# ============================================================================
# تابع اصلی برازش همه توزیع‌ها
# ============================================================================
@njit
def fit_all_distributions_numba(data):
    out = np.full(33, np.nan, dtype=np.float64)
    n = len(data)
    if n < MIN_VALID_VALUES:
        out[0] = -1
        return out

    data_clean = data[~np.isnan(data)]
    n_clean = len(data_clean)
    if n_clean < MIN_VALID_VALUES:
        out[0] = -1
        out[28] = np.nanmean(data)
        out[29] = np.nanstd(data)
        out[30] = 0.0
        out[31] = np.nanmedian(data)
        out[32] = n_clean
        return out

    mean, std, skew, median, count = compute_stats_numba(data_clean)
    if std < 1e-10:
        out[0] = -1
        out[28] = mean
        out[29] = std
        out[30] = skew
        out[31] = median
        out[32] = count
        return out

    # برازش ۴ توزیع
    n_p1, n_p2, n_ll, n_aicc, n_bic, n_k = fit_normal_full_numba(data_clean)
    s_p1, s_p2, s_p3, s_ll, s_aicc, s_bic, s_k = fit_skewnorm_full_numba(data_clean)
    g_p1, g_p2, g_p3, g_ll, g_aicc, g_bic, g_k = fit_gev_full_numba(data_clean)
    p_p1, p_p2, p_p3, p_ll, p_aicc, p_bic, p_k = fit_pearson3_full_numba(data_clean)

    # انتخاب بهترین
    best_code = -1
    best_aicc = np.inf
    if not np.isnan(n_aicc) and n_aicc < best_aicc:
        best_aicc = n_aicc
        best_code = 0
    if not np.isnan(s_aicc) and s_aicc < best_aicc:
        best_aicc = s_aicc
        best_code = 1
    if not np.isnan(g_aicc) and g_aicc < best_aicc:
        best_aicc = g_aicc
        best_code = 2
    if not np.isnan(p_aicc) and p_aicc < best_aicc:
        best_aicc = p_aicc
        best_code = 3

    # پر کردن خروجی
    idx = 0
    out[idx] = best_code
    idx += 1
    out[idx] = n_p1
    idx += 1
    out[idx] = n_p2
    idx += 1
    out[idx] = n_ll
    idx += 1
    out[idx] = n_aicc
    idx += 1
    out[idx] = n_bic
    idx += 1
    out[idx] = n_k
    idx += 1
    out[idx] = s_p1
    idx += 1
    out[idx] = s_p2
    idx += 1
    out[idx] = s_p3
    idx += 1
    out[idx] = s_ll
    idx += 1
    out[idx] = s_aicc
    idx += 1
    out[idx] = s_bic
    idx += 1
    out[idx] = s_k
    idx += 1
    out[idx] = g_p1
    idx += 1
    out[idx] = g_p2
    idx += 1
    out[idx] = g_p3
    idx += 1
    out[idx] = g_ll
    idx += 1
    out[idx] = g_aicc
    idx += 1
    out[idx] = g_bic
    idx += 1
    out[idx] = g_k
    idx += 1
    out[idx] = p_p1
    idx += 1
    out[idx] = p_p2
    idx += 1
    out[idx] = p_p3
    idx += 1
    out[idx] = p_ll
    idx += 1
    out[idx] = p_aicc
    idx += 1
    out[idx] = p_bic
    idx += 1
    out[idx] = p_k
    idx += 1
    out[idx] = mean
    idx += 1
    out[idx] = std
    idx += 1
    out[idx] = skew
    idx += 1
    out[idx] = median
    idx += 1
    out[idx] = count

    return out

# ============================================================================
# Wrapper
# ============================================================================
def fit_distribution(values):
    if values is None or len(values) < MIN_VALID_VALUES:
        return None
    return fit_all_distributions_numba(values)

if __name__ == "__main__":
    sample = np.random.randn(200).astype(np.float64)
    _ = fit_all_distributions_numba(sample)
    print("✅ Numba distributions compiled successfully.")