import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def compute_basic_metrics(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return dict(bias=np.nan, rmse=np.nan, mae=np.nan, r=np.nan, r2=np.nan, nse=np.nan)
    o = obs[mask]
    s = sim[mask]
    bias = np.mean(s - o)
    rmse = np.sqrt(np.mean((s - o) ** 2))
    mae = np.mean(np.abs(s - o))
    r = np.corrcoef(o, s)[0, 1] if o.size > 1 else np.nan
    r2 = r ** 2 if not np.isnan(r) else np.nan
    nse = 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2) if np.sum((o - o.mean()) ** 2) > 0 else np.nan
    return dict(bias=bias, rmse=rmse, mae=mae, r=r, r2=r2, nse=nse)


def compute_kge(obs, sim):
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    o = obs[mask]
    s = sim[mask]
    r = np.corrcoef(o, s)[0, 1]
    alpha = np.std(s) / np.std(o) if np.std(o) > 0 else np.nan
    beta = np.mean(s) / np.mean(o) if np.mean(o) > 0 else np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def validate_datasets(ds_obs, ds_sim, time_dim='time'):
    metrics = {}
    metrics['bias'] = xr.apply_ufunc(
        lambda o, s: np.nanmean(s - o), ds_obs, ds_sim,
        input_core_dims=[[time_dim], [time_dim]],
        vectorize=True, dask='parallelized', output_dtypes=[float]
    )
    metrics['rmse'] = xr.apply_ufunc(
        lambda o, s: np.sqrt(np.nanmean((s - o) ** 2)), ds_obs, ds_sim,
        input_core_dims=[[time_dim], [time_dim]],
        vectorize=True, dask='parallelized', output_dtypes=[float]
    )
    metrics['nse'] = xr.apply_ufunc(
        lambda o, s: 1 - np.sum((s - o) ** 2) / np.sum((o - np.mean(o)) ** 2) if np.sum((o - np.mean(o)) ** 2) > 0 else np.nan,
        ds_obs, ds_sim,
        input_core_dims=[[time_dim], [time_dim]],
        vectorize=True, dask='parallelized', output_dtypes=[float]
    )
    return xr.Dataset(metrics)


def extremes_metrics(ds, time_dim='time'):
    rx1 = ds.max(dim=time_dim)
    p95 = ds.quantile(0.95, dim=time_dim)
    p99 = ds.quantile(0.99, dim=time_dim)
    return dict(rx1=rx1, p95=p95, p99=p99)


def quantile_map_single_series(obs, sim, quantiles=np.linspace(0, 1, 101)):
    q_obs = np.nanquantile(obs, quantiles)
    q_sim = np.nanquantile(sim, quantiles)

    def map_values(x):
        return np.interp(x, q_sim, q_obs, left=q_obs[0], right=q_obs[-1])

    return map_values
