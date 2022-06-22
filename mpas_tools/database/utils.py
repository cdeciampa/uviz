import os
import glob
import numpy as np
import xarray as xr
import metpy.calc as mpcalc

def choose_forecast(pathfolder, date='20180911', regridded=False, init_time='0Z'):
    # 0Z chosen by default. 12Z can also be chosen.
    
    # 00000 files has 9/11 0Z through 9/16 0Z
    # 43200 files has 9/11 12Z through 9/16 12Z
    if init_time == '0Z':
        file_ext = '00000'
        datefolder = os.path.join(pathfolder, date+'00')
    elif init_time == '12Z':
        file_ext = '43200'
        datefolder = os.path.join(pathfolder, date+'12')
    else:
        raise ValueError("Enter '0Z' or '12Z' for forecast initialization time.")
    
    # Selects files that have been regridded
    if regridded == True:
        files = glob.glob(os.path.join(datefolder, f'**/*{file_ext}.nc_regrid.nc'), recursive=True)
    elif regridded == False:
        files = glob.glob(os.path.join(datefolder, f'**/*[!_regrid]{file_ext}.nc'), recursive=True)
    else:
        raise ValueError("'Regridded' is a boolean value.")
    
    # Separates regridded files into h1, h2, and h4 files 
    h1_files = [f for f in files if 'h1' in f]
    h2_files = [f for f in files if 'h2' in f]
    h4_files = [f for f in files if 'h4' in f]
    
    # Opens files into separate datasets
    h1_ds = xr.open_mfdataset(h1_files)
    h2_ds = xr.open_mfdataset(h2_files)
    h4_ds = xr.open_mfdataset(h4_files)
    
    return h1_ds, h2_ds, h4_ds

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def choose_level(ds, height):
    """
    Converts height in [m] to pressure level in [hPa] and 
    returns indexed value in [m]
    """
    
    pressure_hPa = mpcalc.height_to_pressure_std(ds.lev)
    pressure_hPa = pressure_hPa.round(0)
    idx = find_nearest(pressure_hPa, height)

    return idx