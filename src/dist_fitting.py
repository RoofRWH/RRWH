"""Rainfall distribution fitting helpers for the public RRWH workflow."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.special import beta as beta_fn, betainc, betaincinv
from scipy.stats import rv_continuous

warnings.filterwarnings('ignore')

FIT_DISTRIBUTIONS = [
    'lognorm', 'gamma', 'gengamma', 'gb2', 'betaprime', 'burr', 'burr12', 'dagum', 'lomax', 'fisk',
    'genpareto', 'genlogistic', 'loggamma', 'loglaplace', 'halfnorm', 'levy', 'gumbel_r', 'genextreme',
    'norm', 'pareto', 'pearson3', 'weibull_min', 'invweibull', 'expon', 'exponweibull', 'genexpon', 'geninvgauss',
    'invgauss', 'fatiguelife', 'gennorm', 'kappa4', 'skewnorm'
]

DIST_FIT_KWARGS = {
    'gb2': {'floc': 0},
    'dagum': {'floc': 0},
    'lognorm': {'floc': 0},
    'gamma': {'floc': 0},
    'gengamma': {'floc': 0},
    'betaprime': {'floc': 0},
    'burr': {'floc': 0},
    'burr12': {'floc': 0},
    'lomax': {'floc': 0},
    'fisk': {'floc': 0},
    'genpareto': {'floc': 0},
    'loggamma': {'floc': 0},
    'loglaplace': {'floc': 0},
    'weibull_min': {'floc': 0},
    'invweibull': {'floc': 0},
    'expon': {'floc': 0},
    'exponweibull': {'floc': 0},
    'genexpon': {'floc': 0},
    'geninvgauss': {'floc': 0},
    'invgauss': {'floc': 0},
    'fatiguelife': {'floc': 0},
    'pareto': {'floc': 0},
    'pearson3': {'floc': 0},
}


class gb2_gen(rv_continuous):
    shapes = 'alpha, p, q'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('a', 0.0)
        super().__init__(*args, **kwargs)

    def _argcheck(self, alpha, p, q):
        return (alpha > 0) & (p > 0) & (q > 0)

    def _pdf(self, x, alpha, p, q):
        x = np.asarray(x, dtype=float)
        pdf = np.zeros_like(x, dtype=float)
        mask = x > 0
        if np.any(mask):
            xa = np.power(x[mask], alpha)
            numerator = alpha * np.power(x[mask], alpha * p - 1.0)
            denominator = beta_fn(p, q) * np.power(1.0 + xa, p + q)
            pdf[mask] = numerator / denominator
        return pdf

    def _cdf(self, x, alpha, p, q):
        x = np.asarray(x, dtype=float)
        cdf = np.zeros_like(x, dtype=float)
        mask = x > 0
        if np.any(mask):
            xa = np.power(x[mask], alpha)
            z = np.clip(xa / (1.0 + xa), 0.0, 1.0)
            cdf[mask] = betainc(p, q, z)
        return cdf

    def _ppf(self, probabilities, alpha, p, q):
        probabilities = np.asarray(probabilities, dtype=float)
        result = np.zeros_like(probabilities, dtype=float)
        lower = probabilities <= 0.0
        upper = probabilities >= 1.0
        mid = (~lower) & (~upper)
        result[lower] = 0.0
        if np.any(mid):
            z = betaincinv(p, q, np.clip(probabilities[mid], 1e-12, 1 - 1e-12))
            ratio = np.clip(z / (1.0 - z), 0.0, None)
            result[mid] = np.power(ratio, 1.0 / alpha)
        result[upper] = np.inf
        return result


class dagum_gen(rv_continuous):
    shapes = 'alpha, p'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('a', 0.0)
        super().__init__(*args, **kwargs)

    def _argcheck(self, alpha, p):
        return (alpha > 0) & (p > 0)

    def _pdf(self, x, alpha, p):
        x = np.asarray(x, dtype=float)
        pdf = np.zeros_like(x, dtype=float)
        mask = x > 0
        if np.any(mask):
            xa = np.power(x[mask], alpha)
            pdf[mask] = (alpha * p * np.power(x[mask], alpha * p - 1.0)) / np.power(1.0 + xa, p + 1.0)
        return pdf

    def _cdf(self, x, alpha, p):
        x = np.asarray(x, dtype=float)
        cdf = np.zeros_like(x, dtype=float)
        mask = x > 0
        if np.any(mask):
            cdf[mask] = np.power(1.0 + np.power(x[mask], -alpha), -p)
        return cdf

    def _ppf(self, probabilities, alpha, p):
        probabilities = np.asarray(probabilities, dtype=float)
        result = np.zeros_like(probabilities, dtype=float)
        lower = probabilities <= 0.0
        upper = probabilities >= 1.0
        mid = (~lower) & (~upper)
        result[lower] = 0.0
        if np.any(mid):
            ratio = np.power(np.clip(probabilities[mid], 1e-12, 1 - 1e-12), -1.0 / p) - 1.0
            ratio = np.clip(ratio, 1e-12, None)
            result[mid] = np.power(ratio, -1.0 / alpha)
        result[upper] = np.inf
        return result


GB2 = gb2_gen(name='gb2', a=0.0)
DAGUM = dagum_gen(name='dagum', a=0.0)
DIST_REGISTRY = {
    'gb2': GB2,
    'dagum': DAGUM,
    'exponweibull': st.exponweib,
}


def get_distribution(dist_name: str):
    key = str(dist_name).lower()
    if key in DIST_REGISTRY:
        return DIST_REGISTRY[key]
    if hasattr(st, key):
        return getattr(st, key)
    if hasattr(st, dist_name):
        return getattr(st, dist_name)
    return None


def _filter_valid_data(data: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(data), dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    return values


def fit_single_distribution(data: Iterable[float], dist_name: str) -> Dict:
    valid_data = _filter_valid_data(data)
    if valid_data.size < 10:
        return {
            'distribution': dist_name,
            'status': 'failed',
            'error': 'not enough wet-day observations',
            'aic': np.inf,
        }

    dist = get_distribution(dist_name)
    if dist is None:
        return {
            'distribution': dist_name,
            'status': 'failed',
            'error': 'distribution not available',
            'aic': np.inf,
        }

    fit_kwargs = DIST_FIT_KWARGS.get(dist_name.lower(), {})
    try:
        params = dist.fit(valid_data, **fit_kwargs)
        log_likelihood = np.sum(dist.logpdf(valid_data, *params))
        k = len(params)
        n = len(valid_data)
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood
        ks_stat, ks_p = st.kstest(valid_data, lambda x: dist.cdf(x, *params))
        quantile_probs = [0.10, 0.50, 0.90]
        quantiles = dist.ppf(quantile_probs, *params)
        return {
            'distribution': dist_name,
            'params': tuple(float(x) for x in params),
            'aic': float(aic),
            'bic': float(bic),
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'p10': float(quantiles[0]),
            'p50': float(quantiles[1]),
            'p90': float(quantiles[2]),
            'n': int(n),
            'status': 'success',
        }
    except Exception as exc:
        return {
            'distribution': dist_name,
            'status': 'failed',
            'error': str(exc),
            'aic': np.inf,
        }


def fit_all_distributions(data: Iterable[float], dist_list: Optional[Iterable[str]] = None) -> List[Dict]:
    dist_list = list(dist_list or FIT_DISTRIBUTIONS)
    return [fit_single_distribution(data, dist_name) for dist_name in dist_list]


def select_best_distribution(results: List[Dict]) -> Optional[Dict]:
    successful = [result for result in results if result.get('status') == 'success']
    if not successful:
        return None
    return sorted(successful, key=lambda item: item['aic'])[0]


def rank_distributions(data: Iterable[float], dist_list: Optional[Iterable[str]] = None) -> pd.DataFrame:
    results = fit_all_distributions(data, dist_list=dist_list)
    df = pd.DataFrame(results)
    if df.empty:
        return df
    success_df = df[df['status'] == 'success'].copy()
    if success_df.empty:
        return success_df
    success_df = success_df.sort_values('aic').reset_index(drop=True)
    success_df['delta_aic'] = success_df['aic'] - success_df['aic'].min()
    with np.errstate(over='ignore'):
        weights = np.exp(-0.5 * success_df['delta_aic'].to_numpy(dtype=float))
    weight_sum = float(np.nansum(weights))
    success_df['aic_weight'] = weights / weight_sum if weight_sum > 0 else np.nan
    success_df['rank_aic'] = np.arange(1, len(success_df) + 1)
    return success_df
