#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
numerical_engine/distributions.py - برازش توزیع‌ها با معماری پویا
"""

import numpy as np
from numba import njit
import math
from constants import MIN_VALID_VALUES
from zarr_schema import DISTRIBUTIONS

# ============================================================================
# توابع پایه Numba
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
# توزیع Normal
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
    loglik = np.sum(logpdf_normal_numba(data, mean, std))
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
    alpha = max(-10.0, min(10.0, skewness_to_alpha(skew)))
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
# توزیع Bimodal (با استفاده از ماژول خارجی)
# ============================================================================
def fit_bimodal_normal(data):
    """برازش Bimodal با استفاده از bimodal_normal.py"""
    try:
        from bimodal_normal import BimodalNormal
        model = BimodalNormal.fit(data, n_components=2, random_state=42, silent=True, n_jobs=-1)
        w1 = model.w1
        mu1 = model.mu1
        sigma1 = model.sigma1
        mu2 = model.mu2
        sigma2 = model.sigma2
        loglik = model.log_likelihood(data)
        n = len(data)
        k = 5
        aicc = compute_aicc_numba(loglik, k, n)
        bic = compute_bic_numba(loglik, k, n)
        return w1, mu1, sigma1, mu2, sigma2, loglik, aicc, bic
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

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
        a = np.array([2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637])
        b = np.array([-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833])
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
    else:
        r = p if y > 0 else 1-p
        s = math.log(-math.log(r))
        c = np.array([0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                      0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
                      0.0000321767881768, 0.0000002888167364, 0.0000003960315187])
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
def fit_pearson3_full_numba(data):
    n = len(data)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    # محاسبه چولگی
    skew = np.mean(((data - mean) / std) ** 3)
    skew = max(-2.999, min(2.999, skew))
    # پیاده‌سازی ساده برای Pearson III (در صورت نیاز کامل‌تر کنید)
    # اینجا یک نسخه ساده برای نمونه ارائه شده است
    if abs(skew) < 0.1:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    # تخمین پارامترها با روش گشتاورها
    shape = 4 / (skew ** 2) if skew != 0 else 10
    scale = std / math.sqrt(shape) if shape > 0 else std
    loc = mean - shape * scale
    if scale <= 0 or shape <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    # محاسبه log-likelihood (تقریبی)
    loglik = 0.0
    for x in data:
        z = (x - loc) / scale
        if z <= 0:
            loglik = np.nan
            break
        loglik += (shape - 1) * math.log(z) - z - math.lgamma(shape) - math.log(scale)
    if np.isnan(loglik):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3
    aicc = compute_aicc_numba(loglik, 3, n)
    bic = compute_bic_numba(loglik, 3, n)
    return shape, scale, loc, loglik, aicc, bic, 3

# ============================================================================
# دیکشنری توابع برازش
# ============================================================================
FIT_FUNCS = {
    "normal": fit_normal_full_numba,
    "skew": fit_skewnorm_full_numba,
    "bimodal": fit_bimodal_normal,
    "pearson": fit_pearson3_full_numba,
}

# ============================================================================
# تابع اصلی برازش
# ============================================================================
def fit_all_distributions(data):
    """برازش همه توزیع‌ها و بازگرداندن نتایج به صورت دیکشنری"""
    n = len(data)
    out = {}
    if n < MIN_VALID_VALUES:
        out["best_dist"] = -1
        out["mean"] = np.nanmean(data)
        out["std"] = np.nanstd(data)
        out["skewness"] = 0.0
        out["median"] = np.nanmedian(data)
        out["count"] = len(data[~np.isnan(data)])
        return out

    data_clean = data[~np.isnan(data)]
    n_clean = len(data_clean)
    if n_clean < MIN_VALID_VALUES:
        out["best_dist"] = -1
        out["mean"] = np.nanmean(data)
        out["std"] = np.nanstd(data)
        out["skewness"] = 0.0
        out["median"] = np.nanmedian(data)
        out["count"] = n_clean
        return out

    mean, std, skew, median, count = compute_stats_numba(data_clean)
    out["mean"] = mean
    out["std"] = std
    out["skewness"] = skew
    out["median"] = median
    out["count"] = count

    if std < 1e-10:
        out["best_dist"] = -1
        return out

    best_aicc = np.inf
    best_code = -1
    results = {}

    for dist in DISTRIBUTIONS:
        name = dist["name"]
        fit_func = FIT_FUNCS.get(name)
        if fit_func is None:
            continue
        try:
            res = fit_func(data_clean)
            if res is None or np.any(np.isnan(res)):
                continue
            results[name] = res
            # استخراج aicc (آخرین عنصر دوم)
            if len(res) >= 2:
                aicc = res[-2]
                if not np.isnan(aicc) and aicc < best_aicc:
                    best_aicc = aicc
                    best_code = dist["code"]
        except Exception:
            continue

    out["best_dist"] = best_code

    # ذخیره پارامترها
    for dist in DISTRIBUTIONS:
        name = dist["name"]
        if name in results:
            res = results[name]
            for i, (pname, _, _) in enumerate(dist["params"]):
                if i < len(res):
                    out[f"{name}_{pname}"] = res[i]
            out[f"{name}_loglik"] = res[-3] if len(res) >= 3 else np.nan
            out[f"{name}_aicc"] = res[-2] if len(res) >= 2 else np.nan
            out[f"{name}_bic"] = res[-1] if len(res) >= 1 else np.nan
        else:
            for pname, _, _ in dist["params"]:
                out[f"{name}_{pname}"] = np.nan
            out[f"{name}_loglik"] = np.nan
            out[f"{name}_aicc"] = np.nan
            out[f"{name}_bic"] = np.nan

    return out

def fit_distribution(values):
    if values is None or len(values) < MIN_VALID_VALUES:
        return None
    return fit_all_distributions(values)

if __name__ == "__main__":
    print("✅ توزیع‌ها بارگذاری شدند:", list(FIT_FUNCS.keys()))
