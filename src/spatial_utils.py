import numpy as np
import xarray as xr


def get_utm_crs(lon, lat):
    """Calculate the EPSG code for the UTM zone containing the given longitude/latitude."""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f'EPSG:{epsg}'


def mask_raster_by_shape(da, gdf):
    """Mask an xarray DataArray using a GeoDataFrame."""
    da = da.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return da
