#from sklearn.metrics.pairwise import haversine_distances
#from math import radians, degrees, asin, sin, atan2, cos
import numpy as np
import xarray as xr

from haversine import *

from plotting.utils import basin_bboxes

# def calc_point_from_dist(start_coords, distance, bearing, dist_scale='km'):
#     """
#     Calculates and returns a point a given distance away.
#     start_coords :: (tuple or array), give in lon, lat form.
#     distance :: (int or float)
#     bearing :: (int or float), give in degrees
#     dist_scale (optional)

#     """
    
#     # Derives Earth's radius based on given length scale.
#     if dist_scale == 'miles' or dist_scale == 'mi':
#         radius = 3958.7603
#     elif dist_scale == 'kilometers' or dist_scale == 'km':
#         radius = 6371.0072
#     elif dist_scale == 'meters' or dist_scale == 'm':
#         radius = (6371.0072)*10E5
#     else:
#         raise ValueError('Supply distance in miles, kilometers, or meters.')

#     # Converts degrees to radians
#     lon1, lat1 = start_coords
#     bearing = radians(bearing)
#     lat1 = radians(lat1)
#     lon1 = radians(lon1)

#     # Calculates endpoint based on distance (modified Haversine)
#     lat2 = asin((sin(lat1)*cos(distance/radius))+\
#                 (cos(lat1)*sin(distance/radius)*cos(bearing)))
#     lon2 = lon1+atan2(sin(bearing)*sin(distance/radius)*cos(lat1), 
#                       cos(distance/radius)-(sin(lat1)*sin(lat2)))
    
#     # Converts radians back to degrees
#     lat2 = degrees(lat2)
#     lon2 = degrees(lon2)

#     return (lon2, lat2)

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
    
    # Spread for target bbox
    # lon_e = calc_point_from_dist((lon, lat), distance=center_dist, bearing=90, dist_scale='km')[0]
    # lon_w = calc_point_from_dist((lon, lat), distance=center_dist, bearing=270, dist_scale='km')[0]
    # lat_n = calc_point_from_dist((lon, lat), distance=center_dist, bearing=0, dist_scale='km')[1]
    # lat_s = calc_point_from_dist((lon, lat), distance=center_dist, bearing=180, dist_scale='km')[1]
    
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