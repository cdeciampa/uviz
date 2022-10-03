#from sklearn.metrics.pairwise import haversine_distances
#from math import radians, degrees, asin, sin, atan2, cos
import numpy as np
import xarray as xr

from haversine import *

from uviz.plotting.utils import basin_bboxes

def find_TC_bbox(ds, basin, time, center_dist=100, unit='km'):
    """
    Tracks tropical cyclones based on basin, specific humidity, 
    and minimum SLP.
    """
    
    # Slice data by specified date/time
    data = ds.isel(time=time)
    
    # Slice data by lat/lon bounding (specific to basin)
    lons, lats = basin_bboxes(basin)
    data = data.where((data.lon >= lons[0]) & (data.lon <= lons[1]) &\
                      (data.lat >= lats[0]) & (data.lat <= lats[1]), drop=True)
    
    # Slice data where surface specific humidity > 2% or total PW >= 74
    try:
        data = data.where(data.Q.isel(lev=-1) >= 0.02, drop=True)
    except AttributeError:
        data = data.where(data.TMQ >= 74.0, drop=True)
    
    # Slice data based on minimum sea level pressure value
    data = data.where(data.PSL == data.PSL.values.min(), drop=True)
    
    # Coordinates for center of TC
    lat = data['lat'].values[0]
    lon = data['lon'].values[0]
    
    # Normalizes coords to [-180, 180] and [-90, 90]
    #lat, lon = _normalize(lat, lon)
    
    # Spread for target bbox
    lon_e = inverse_haversine((lat, lon), center_dist, Direction.EAST, unit=unit)[1]
    lon_w = inverse_haversine((lat, lon), center_dist, Direction.WEST, unit=unit)[1]
    lat_n = inverse_haversine((lat, lon), center_dist, Direction.NORTH, unit=unit)[0]
    lat_s = inverse_haversine((lat, lon), center_dist, Direction.SOUTH, unit=unit)[0]
    
    # Back to [0, 360] for longitude
    # if lon_e < 0:
    #     lon_e += 360 
    # if lon_w < 0:
    #     lon_w += 360
    
    lon_range = (lon_w, lon_e)
    lat_range = (lat_s, lat_n)
    
    return lon_range, lat_range


def sfc_wind_corr(z_r=64.0, constant=None, wind_factor=None, alpha=0.11, z_0 = 0.0002):
    """
    Function to correct lowest level model wind speed to 10m wind speed.
    
    Parameters
    -------
    z_r : optional, float
        The height of the lowest model level. If negative, uses CAM default.
        
    constant : float
        The corresponding constant for either power/log correction.
        
    wind_factor : str
        Defines type of correction as either 'power' or 'log'.
    
    alpha : optional, float
    
    z_0 : optional, float
        Roughness coefficient, default is 0.0002, corresponding to open ocean.
    
    Raises
    ------
    NotImplementedError
        If only one of wind_factor or constant are assigned.
    ValueError
        If wind_factor is incorrectly assigned.
        
    Returns
    --------
    factor : float
        10 m wind speed correction factor.
    """
    
    if constant is not None and wind_factor is None:
        raise NotImplementedError("Can't assign a constant without a wind_factor, must assign both.")
    
    eps = 1.0E-8 # "Epsilon" value - approaches 0
    # if wind_factor is not None:
        # print(f'SURFACEWINDCORRFACTOR: Getting sfc wind correction factor using: {wind_factor} technique.')
    
    if z_r < 0.0:
        z_r = 64.0
        # print(f'SURFACEWINDCORRFACTOR: Using CAM default lowest model level of: {z_r} m.')
    
    if wind_factor == 'power':
        z_10 = 10.0
        z_r = 64.0
        if (constant is not None) and (constant > eps):
            alpha = constant
        factor = (z_10/z_r)**alpha
        # print(f'SURFACEWINDCORRFACTOR: Using factor: {factor}.')
    elif wind_factor == 'log':
        # Garratt 1992 -- Wind profile formulation
        # Wieringa 1993 -- roughness coefficient
        z_10 = 10.0
        if (constant is not None) and (constant > eps):
            z_0 = constant # roughness coefficient (length)
        factor = 1 + (np.log(z_10/z_r)/np.log(z_r/z_0))
    elif wind_factor is None and constant is None:
        # print('SURFACEWINDCORRFACTOR: No correction used.')
        factor = 1.0
    else:
        raise ValueError('SURFACEWINDCORRFACTOR: Incorrect wind correction type. Must assign "power" or "log" if assigning wind_factor.')
    
    return factor
    