import os

import numpy as np
import rioxarray
import xarray as xr
import yaml

try:
    import xesmf as xe
except ImportError:
    xe = None
    print('Warning: xesmf not found. Regridding functions requiring xesmf will fail.')


def load_config(config_path='config/parameters.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_paths(paths_path='config/paths.yaml'):
    with open(paths_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_chirps(path):
    """Load CHIRPS dataset."""
    if os.path.isdir(path):
        ds = xr.open_mfdataset(os.path.join(path, '*.nc'), combine='by_coords', parallel=True)
    else:
        ds = xr.open_dataset(path)

    if 'precip' in ds:
        return ds['precip']
    return ds


def load_geotiff(path):
    """Load GeoTIFF using rioxarray."""
    da = rioxarray.open_rasterio(path, masked=True)
    if 'band' in da.dims and da.sizes['band'] == 1:
        da = da.squeeze('band', drop=True)
    return da


def align_to_reference(da, reference_da):
    """Reproject and align a DataArray to match a reference DataArray."""
    if da.rio.crs is None:
        da.rio.write_crs('EPSG:4326', inplace=True)
    if reference_da.rio.crs is None:
        raise ValueError('Reference DataArray must have a CRS.')
    return da.rio.reproject_match(reference_da)


def reproject_raster(da, target_crs):
    """Reproject xarray DataArray to target CRS."""
    if da.rio.crs is None:
        da.rio.write_crs('EPSG:4326', inplace=True)
    return da.rio.reproject(target_crs)


def align_resolution(high_res_da, low_res_da, method='bilinear'):
    """Align resolution of two DataArrays."""
    return low_res_da.interp_like(high_res_da, method=method)


def compute_rainfall_stats(daily_rain):
    """Compute annual sum, skewness, and 95th percentile."""
    annual_rain = daily_rain.groupby('time.year').sum(dim='time').mean(dim='year')

    try:
        rain_skewness = daily_rain.skew(dim='time')
    except AttributeError:
        mean = daily_rain.mean(dim='time')
        std = daily_rain.std(dim='time')
        m3 = ((daily_rain - mean) ** 3).mean(dim='time')
        rain_skewness = m3 / (std ** 3)

    if daily_rain.chunks is not None:
        chunk_spec = {d: 'auto' for d in daily_rain.dims if d != 'time'}
        chunk_spec['time'] = -1
        rain_heaviness = daily_rain.chunk(chunk_spec).quantile(0.95, dim='time')
    else:
        rain_heaviness = daily_rain.quantile(0.95, dim='time')

    return annual_rain, rain_skewness, rain_heaviness


def clean_data(da):
    """Remove negative values and handle NaNs."""
    return da.where(da >= 0, np.nan)


def regrid_to_target(ds_src: xr.Dataset, ds_target_template: xr.Dataset, method='conservative'):
    """Regrid ds_src to the grid of ds_target_template."""
    if xe is None:
        raise ImportError('xesmf is required for this function but not installed.')

    regridder = xe.Regridder(ds_src, ds_target_template, method, periodic=False, reuse_weights=True)
    ds_out = regridder(ds_src)
    regridder.clean_weight_file()
    return ds_out
