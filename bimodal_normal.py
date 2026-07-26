#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bimodal_normal.py

برازش توزیع نرمال دو قله‌ای (Bimodal Normal Distribution)
با استفاده از Gaussian Mixture Model

نسخه ۳.۱ - اصلاحات نهایی برای انتشار در PyPI و ارجاع علمی

ویژگی‌ها:
- fit فقط n_components=1 یا 2 می‌پذیرد
- covariance_type='tied' مدیریت می‌شود
- ppf با محدوده پویا
- cached_property برای overlap_coefficient
- Hartigan Dip Test فقط با diptest (بدون fallback)
- save/load متقارن (pickle و json)
- ذخیره AIC, BIC, log_likelihood
- Type Hint با numpy.typing
- تست‌های واحد (pytest)

نویسنده: A. Kazemi
آخرین بروزرسانی: ۱۴۰۴/۰۵/۰۳
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import kurtosis, skew
from dataclasses import dataclass, asdict
from typing import Optional, Union, Tuple, Dict, List, Any
import warnings
import pickle
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import cached_property

warnings.filterwarnings("ignore")

# ============================================================
# شاخص‌های استاندارد دو قله‌ای
# ============================================================

def ashman_d(mu1: float, mu2: float, sigma1: float, sigma2: float) -> float:
    """شاخص Ashman's D برای تشخیص دو قله‌ای"""
    delta = abs(mu2 - mu1)
    return delta / np.sqrt((sigma1**2 + sigma2**2) / 2)

def bimodality_coefficient_from_moments(skewness: float, kurtosis: float, n: int) -> float:
    """ضریب دو قله‌ای (BC) از گشتاورهای تحلیلی"""
    if n <= 3:
        return (skewness**2 + 1) / (kurtosis + 1)
    kurt_adjusted = kurtosis + 3 * ((n - 1)**2 / ((n - 2) * (n - 3)))
    return (skewness**2 + 1) / (kurt_adjusted + 1e-10)

def hartigan_dip_test(data: npt.NDArray[np.floating]) -> Tuple[float, float]:
    """آزمون دوقله‌ای Hartigan با استفاده از diptest"""
    try:
        import diptest
        result = diptest.diptest(data)
        # بسته به نسخه، ممکن است تاپل یا شیء برگرداند
        if isinstance(result, tuple):
            dip, p_value = result
        else:
            dip = result.dip
            p_value = result.p_value
        return dip, p_value
    except ImportError:
        raise ImportError("برای اجرای آزمون Hartigan Dip، کتابخانه diptest نیاز است. "
                          "با `pip install diptest` آن را نصب کنید.")

# ============================================================
# Dataclass مدل
# ============================================================

@dataclass
class BimodalModelParams:
    w1: float
    w2: float
    mu1: float
    mu2: float
    sigma1: float
    sigma2: float
    n_components: int = 2
    covariance_type: str = "full"
    converged: bool = True
    aic: Optional[float] = None
    bic: Optional[float] = None
    log_likelihood: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================================
# کلاس اصلی
# ============================================================

class BimodalNormal:
    """توزیع نرمال دو قله‌ای (Mixture of two Gaussians)"""
    
    def __init__(
        self,
        w1: float = 0.5,
        mu1: float = 0.0,
        mu2: float = 1.0,
        sigma1: float = 1.0,
        sigma2: float = 1.0,
        covariance_type: str = "full"
    ):
        self.covariance_type = covariance_type
        self.w1 = np.clip(w1, 0.0, 1.0)
        self.w2 = 1.0 - self.w1
        self.sigma1 = max(sigma1, 1e-6)
        self.sigma2 = max(sigma2, 1e-6)
        
        if mu1 < mu2:
            self.mu1, self.mu2 = mu1, mu2
            self._w1, self._w2 = self.w1, self.w2
            self._sigma1, self._sigma2 = self.sigma1, self.sigma2
        else:
            self.mu1, self.mu2 = mu2, mu1
            self._w1, self._w2 = self.w2, self.w1
            self._sigma1, self._sigma2 = self.sigma2, self.sigma1
        
        self.w1, self.w2 = self._w1, self._w2
        self.sigma1, self.sigma2 = self._sigma1, self._sigma2
        
        self._dist1 = norm(self.mu1, self.sigma1)
        self._dist2 = norm(self.mu2, self.sigma2)
        self._gmm: Optional[GaussianMixture] = None
        self._params: Optional[BimodalModelParams] = None
    
    # ============================================================
    # توابع اصلی توزیع
    # ============================================================
    
    def pdf(self, x: Union[npt.NDArray[np.floating], float, list]) -> npt.NDArray[np.floating]:
        x = np.asarray(x)
        return self.w1 * self._dist1.pdf(x) + self.w2 * self._dist2.pdf(x)
    
    def logpdf(self, x: Union[npt.NDArray[np.floating], float, list]) -> npt.NDArray[np.floating]:
        x = np.asarray(x)
        log_w = np.log([self.w1, self.w2])
        log_pdf1 = self._dist1.logpdf(x)
        log_pdf2 = self._dist2.logpdf(x)
        log_probs = np.column_stack([log_pdf1, log_pdf2])
        return logsumexp(log_probs + log_w, axis=1)
    
    def cdf(self, x: Union[npt.NDArray[np.floating], float, list]) -> npt.NDArray[np.floating]:
        x = np.asarray(x)
        return self.w1 * self._dist1.cdf(x) + self.w2 * self._dist2.cdf(x)
    
    def ppf(self, q: Union[npt.NDArray[np.floating], float, list]) -> Union[npt.NDArray[np.floating], float]:
        q = np.asarray(q)
        scalar_input = q.ndim == 0 or len(q) == 1
        
        # محدوده پویا از صدک‌های مؤلفه‌ها
        x_low = min(
            self._dist1.ppf(0.001),
            self._dist2.ppf(0.001)
        ) - 1.0
        x_high = max(
            self._dist1.ppf(0.999),
            self._dist2.ppf(0.999)
        ) + 1.0
        
        def solve_one(prob: float) -> float:
            return brentq(lambda x: self.cdf(x) - prob, x_low, x_high)
        
        if scalar_input:
            return solve_one(float(q))
        else:
            return np.array([solve_one(p) for p in q])
    
    def sample(self, n_samples: int = 1000, random_state: Optional[int] = None) -> npt.NDArray[np.floating]:
        rng = np.random.default_rng(random_state)
        components = rng.choice([0, 1], size=n_samples, p=[self.w1, self.w2])
        n1 = np.sum(components == 0)
        n2 = n_samples - n1
        samples1 = rng.normal(self.mu1, self.sigma1, n1)
        samples2 = rng.normal(self.mu2, self.sigma2, n2)
        samples = np.concatenate([samples1, samples2])
        rng.shuffle(samples)
        return samples
    
    # ============================================================
    # شاخص‌های دو قله‌ای
    # ============================================================
    
    @property
    def peak_separation(self) -> float:
        return self.mu2 - self.mu1
    
    @property
    def weight_imbalance(self) -> float:
        return abs(self.w1 - self.w2)
    
    @property
    def max_tail_width(self) -> float:
        return max(self.sigma1, self.sigma2)
    
    @property
    def bimodality_strength(self) -> float:
        return (self.w1 * self.w2 * self.peak_separation**2) / (self.sigma1 * self.sigma2)
    
    @cached_property
    def overlap_coefficient(self) -> float:
        f1 = lambda x: self._dist1.pdf(x)
        f2 = lambda x: self._dist2.pdf(x)
        return quad(lambda x: np.minimum(f1(x), f2(x)), -np.inf, np.inf)[0]
    
    @property
    def ashman_d(self) -> float:
        return ashman_d(self.mu1, self.mu2, self.sigma1, self.sigma2)
    
    @property
    def bimodality_coefficient(self) -> float:
        return bimodality_coefficient_from_moments(
            self.skewness,
            self.kurtosis,
            n=10000
        )
    
    # ============================================================
    # گشتاورهای تحلیلی
    # ============================================================
    
    @property
    def mean(self) -> float:
        return self.w1 * self.mu1 + self.w2 * self.mu2
    
    @property
    def variance(self) -> float:
        mu = self.mean
        var1 = self.sigma1**2 + (self.mu1 - mu)**2
        var2 = self.sigma2**2 + (self.mu2 - mu)**2
        return self.w1 * var1 + self.w2 * var2
    
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)
    
    @property
    def skewness(self) -> float:
        mu = self.mean
        sigma = self.std
        s1 = self.w1 * ((self.mu1 - mu)**3 + 3 * (self.mu1 - mu) * self.sigma1**2)
        s2 = self.w2 * ((self.mu2 - mu)**3 + 3 * (self.mu2 - mu) * self.sigma2**2)
        return (s1 + s2) / sigma**3
    
    @property
    def kurtosis(self) -> float:
        mu = self.mean
        sigma = self.std
        k1 = self.w1 * ((self.mu1 - mu)**4 + 6 * (self.mu1 - mu)**2 * self.sigma1**2 + 3 * self.sigma1**4)
        k2 = self.w2 * ((self.mu2 - mu)**4 + 6 * (self.mu2 - mu)**2 * self.sigma2**2 + 3 * self.sigma2**4)
        return (k1 + k2) / sigma**4 - 3
    
    # ============================================================
    # معیارهای آماری
    # ============================================================
    
    def log_likelihood(self, x: npt.NDArray[np.floating]) -> float:
        return np.sum(self.logpdf(x))
    
    def score(self, x: npt.NDArray[np.floating]) -> float:
        return np.mean(self.logpdf(x))
    
    def aic(self, x: npt.NDArray[np.floating]) -> float:
        if self._gmm is not None:
            return self._gmm.aic(x.reshape(-1, 1))
        k = 5
        return 2 * k - 2 * self.log_likelihood(x)
    
    def bic(self, x: npt.NDArray[np.floating]) -> float:
        if self._gmm is not None:
            return self._gmm.bic(x.reshape(-1, 1))
        k = 5
        n = len(x)
        return k * np.log(n) - 2 * self.log_likelihood(x)
    
    # ============================================================
    # Bootstrap
    # ============================================================
    
    def confidence_intervals(
        self,
        x: npt.NDArray[np.floating],
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        silent: bool = False
    ) -> Dict[str, Dict[str, float]]:
        n = len(x)
        params = {
            'w1': [], 'mu1': [], 'mu2': [],
            'sigma1': [], 'sigma2': [], 'bimodality_strength': []
        }
        errors = []
        
        for _ in tqdm(range(n_bootstrap), desc="Bootstrap", disable=silent):
            boot_sample = np.random.choice(x, size=n, replace=True)
            try:
                model = BimodalNormal.fit(boot_sample, silent=True)
                params['w1'].append(model.w1)
                params['mu1'].append(model.mu1)
                params['mu2'].append(model.mu2)
                params['sigma1'].append(model.sigma1)
                params['sigma2'].append(model.sigma2)
                params['bimodality_strength'].append(model.bimodality_strength)
            except Exception as e:
                errors.append(str(e))
                continue
        
        if errors and not silent:
            warnings.warn(f"{len(errors)} bootstrap iterations failed.")
        
        ci = {}
        for key, values in params.items():
            if values:
                ci[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    f'{(alpha/2)*100:.1f}%': np.percentile(values, alpha/2 * 100),
                    f'{(1-alpha/2)*100:.1f}%': np.percentile(values, (1 - alpha/2) * 100)
                }
        return ci
    
    # ============================================================
    # برازش
    # ============================================================
    
    @classmethod
    def fit(
        cls,
        data: Union[npt.NDArray[np.floating], pd.Series, list],
        n_components: int = 2,
        random_state: Optional[int] = 42,
        max_iter: int = 1000,
        silent: bool = False,
        covariance_type: str = 'full',
        n_init: int = 10,
        tol: float = 1e-4,
        reg_covar: float = 1e-6,
        init_params: str = 'kmeans'
    ) -> 'BimodalNormal':
        # فقط یک یا دو مؤلفه پشتیبانی می‌شود
        if n_components not in [1, 2]:
            raise ValueError("این کلاس فقط n_components=1 یا 2 را پشتیبانی می‌کند.")
        
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        if len(data) < 10:
            raise ValueError("داده‌ها برای برازش کافی نیستند (حداقل ۱۰ نمونه)")
        
        # حالت تک‌قله‌ای
        if n_components == 1:
            mu = np.mean(data)
            sigma = np.std(data)
            return cls(w1=1.0, mu1=mu, mu2=mu+1e-6, sigma1=sigma, sigma2=1.0, covariance_type=covariance_type)
        
        # حالت دو‌قله‌ای
        gmm = GaussianMixture(n_jobs=-1, 
            n_components=2,
            random_state=random_state,
            max_iter=max_iter,
            covariance_type=covariance_type,
            n_init=n_init,
            tol=tol,
            reg_covar=reg_covar,
            init_params=init_params
        )
        gmm.fit(data.reshape(-1, 1))
        
        if not gmm.converged_ and not silent:
            warnings.warn("EM algorithm did not converge!")
        
        weights = gmm.weights_
        means = gmm.means_.flatten()
        
        # استخراج انحراف معیار بر اساس covariance_type
        if covariance_type == 'diag':
            sigmas = np.sqrt(gmm.covariances_.flatten())
        elif covariance_type == 'full':
            sigmas = np.sqrt(gmm.covariances_[:, 0, 0])
        elif covariance_type == 'tied':
            # در حالت tied، همه مؤلفه‌ها یک σ مشترک دارند
            sigma_common = np.sqrt(gmm.covariances_[0, 0])
            sigmas = np.array([sigma_common, sigma_common])
        else:
            sigmas = np.sqrt(gmm.covariances_.flatten())
        
        # مرتب‌سازی
        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        sigmas = sigmas[order]
        
        model = cls(
            w1=weights[0],
            mu1=means[0],
            mu2=means[1],
            sigma1=sigmas[0],
            sigma2=sigmas[1],
            covariance_type=covariance_type
        )
        
        model._gmm = gmm
        model._params = BimodalModelParams(
            w1=weights[0],
            w2=weights[1],
            mu1=means[0],
            mu2=means[1],
            sigma1=sigmas[0],
            sigma2=sigmas[1],
            n_components=2,
            covariance_type=covariance_type,
            converged=gmm.converged_,
            aic=gmm.aic(data.reshape(-1, 1)),
            bic=gmm.bic(data.reshape(-1, 1)),
            log_likelihood=gmm.score(data.reshape(-1, 1))
        )
        
        return model
    
    # ============================================================
    # Rolling
    # ============================================================
    
    @classmethod
    def fit_rolling(
        cls,
        data: Union[npt.NDArray[np.floating], pd.Series, list],
        window_size: int = 30,
        step: int = 1,
        random_state: Optional[int] = 42,
        silent: bool = False,
        **fit_kwargs
    ) -> pd.DataFrame:
        data = np.asarray(data).flatten()
        results = []
        
        for i in range(0, len(data) - window_size + 1, step):
            window_data = data[i:i + window_size]
            result = {
                'start_idx': i,
                'end_idx': i + window_size,
                'status': 'success',
                'error': None
            }
            try:
                model = cls.fit(window_data, random_state=random_state, silent=silent, **fit_kwargs)
                result.update({
                    'w1': model.w1,
                    'w2': model.w2,
                    'mu1': model.mu1,
                    'mu2': model.mu2,
                    'sigma1': model.sigma1,
                    'sigma2': model.sigma2,
                    'peak_separation': model.peak_separation,
                    'bimodality_strength': model.bimodality_strength,
                    'ashman_d': model.ashman_d,
                    'overlap_coefficient': model.overlap_coefficient,
                    'bimodality_coefficient': model.bimodality_coefficient,
                    'aic': model.aic(window_data) if len(window_data) > 0 else None,
                    'bic': model.bic(window_data) if len(window_data) > 0 else None,
                })
            except Exception as e:
                result['status'] = 'error'
                result['error'] = str(e)
                for key in ['w1', 'w2', 'mu1', 'mu2', 'sigma1', 'sigma2',
                           'peak_separation', 'bimodality_strength',
                           'ashman_d', 'overlap_coefficient', 'bimodality_coefficient',
                           'aic', 'bic']:
                    result[key] = np.nan
            results.append(result)
        
        return pd.DataFrame(results)
    
    # ============================================================
    # انتخاب تعداد مؤلفه‌ها
    # ============================================================
    
    @classmethod
    def select_optimal_components(
        cls,
        data: npt.NDArray[np.floating],
        max_components: int = 5,
        criterion: str = 'bic',
        random_state: int = 42
    ) -> Dict[str, Any]:
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)].reshape(-1, 1)
        
        results = []
        for k in range(1, max_components + 1):
            gmm = GaussianMixture(n_jobs=-1, n_components=k, random_state=random_state)
            gmm.fit(data)
            results.append({
                'n_components': k,
                'AIC': gmm.aic(data),
                'BIC': gmm.bic(data),
                'log_likelihood': gmm.score(data)
            })
        
        df = pd.DataFrame(results)
        best_idx = df[criterion.upper()].idxmin()
        best = df.loc[best_idx]
        
        return {
            'results': df,
            'best': best.to_dict(),
            'n_components_optimal': int(best['n_components']),
            'is_bimodal': best['n_components'] == 2
        }
    
    # ============================================================
    # ذخیره و بارگذاری
    # ============================================================
    
    def save(self, filepath: str) -> None:
        """ذخیره با تشخیص فرمت از پسوند فایل (pickle یا json)"""
        params = {
            'w1': self.w1,
            'w2': self.w2,
            'mu1': self.mu1,
            'mu2': self.mu2,
            'sigma1': self.sigma1,
            'sigma2': self.sigma2,
            'covariance_type': self.covariance_type
        }
        if self._params is not None:
            params['_params'] = self._params.to_dict()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(params, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(params, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BimodalNormal':
        """بارگذاری با تشخیص فرمت از پسوند فایل"""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                params = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
        
        model = cls(
            w1=params['w1'],
            mu1=params['mu1'],
            mu2=params['mu2'],
            sigma1=params['sigma1'],
            sigma2=params['sigma2'],
            covariance_type=params.get('covariance_type', 'full')
        )
        
        if '_params' in params:
            model._params = BimodalModelParams(**params['_params'])
        
        return model
    
    # ============================================================
    # رسم
    # ============================================================
    
    def plot(
        self,
        data: Optional[npt.NDArray[np.floating]] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        x_min = min(self.mu1 - 4 * self.sigma1, self.mu2 - 4 * self.sigma2)
        x_max = max(self.mu1 + 4 * self.sigma1, self.mu2 + 4 * self.sigma2)
        x_grid = np.linspace(x_min, x_max, 200)
        
        if data is not None:
            ax.hist(data, bins=50, density=True, alpha=0.3, color='gray', label='Data')
        
        pdf_vals = self.pdf(x_grid)
        ax.plot(x_grid, pdf_vals, 'r-', linewidth=2, label='Fitted Bimodal')
        
        pdf1 = self.w1 * self._dist1.pdf(x_grid)
        pdf2 = self.w2 * self._dist2.pdf(x_grid)
        ax.plot(x_grid, pdf1, 'b--', linewidth=1.5, label=f'Comp 1 (w={self.w1:.2f})')
        ax.plot(x_grid, pdf2, 'g--', linewidth=1.5, label=f'Comp 2 (w={self.w2:.2f})')
        
        ax.axvline(self.mu1, color='blue', linestyle=':', alpha=0.7, label=f'μ1 = {self.mu1:.3f}')
        ax.axvline(self.mu2, color='green', linestyle=':', alpha=0.7, label=f'μ2 = {self.mu2:.3f}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('Density')
        ax.set_title(f'Bimodal Normal Distribution\nβ={self.bimodality_strength:.3f}, D={self.ashman_d:.3f}, OVL={self.overlap_coefficient:.3f}')
        ax.legend()
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return ax
    
    # ============================================================
    # خلاصه
    # ============================================================
    
    def summary(self, as_dataframe: bool = False) -> Union[Dict[str, Any], Tuple[pd.DataFrame, pd.DataFrame]]:
        parameters = {
            'Parameter': ['w1', 'w2', 'mu1', 'mu2', 'sigma1', 'sigma2'],
            'Value': [self.w1, self.w2, self.mu1, self.mu2, self.sigma1, self.sigma2]
        }
        indicators = {
            'peak_separation': self.peak_separation,
            'weight_imbalance': self.weight_imbalance,
            'max_tail_width': self.max_tail_width,
            'bimodality_strength': self.bimodality_strength,
            'overlap_coefficient': self.overlap_coefficient,
            'ashman_d': self.ashman_d,
            'bimodality_coefficient': self.bimodality_coefficient,
            'mean': self.mean,
            'variance': self.variance,
            'std': self.std,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }
        if as_dataframe:
            df_params = pd.DataFrame(parameters)
            df_indicators = pd.DataFrame(list(indicators.items()), columns=['Indicator', 'Value'])
            return df_params, df_indicators
        return {'parameters': parameters, 'indicators': indicators}
    
    def __repr__(self) -> str:
        return f"BimodalNormal(w1={self.w1:.4f}, w2={self.w2:.4f}, μ1={self.mu1:.4f}, μ2={self.mu2:.4f}, σ1={self.sigma1:.4f}, σ2={self.sigma2:.4f}, β={self.bimodality_strength:.4f}, D={self.ashman_d:.3f})"


# ============================================================
# نسخه و API عمومی
# ============================================================

__version__ = "3.1.0"
__all__ = [
    'BimodalNormal',
    'BimodalModelParams',
    'ashman_d',
    'bimodality_coefficient_from_moments',
    'hartigan_dip_test',
    'select_optimal_components'
]