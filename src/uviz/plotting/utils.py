import numpy as np
import pandas as pd

import xarray as xr

from haversine import inverse_haversine, Direction, Unit
from haversine import haversine_vector as hv

import bokeh.palettes
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class PlotProcessing():

    def spatial_subset(self, ds, bbox, overland=False):
        """
        Function to spatially limit an xr.Dataset.
        
        Parameters
        -------------
        ds   :: xr.Dataset
        bbox :: str or dict, feeds to utils.basin_bboxes
        """
        
        # Retrieves bounding box coordinates
        lons, lats = basin_bboxes(bbox)

        # Subsets data spatially (for regridded data)
        if 'lon' in ds.dims:
            ds = ds.sel(lon=slice(lons[0], lons[1]), lat=slice(lats[0], lats[1]))

        # For native data
        else:
            ds = ds.where((ds['lon'] >= lons[0]) & (ds['lon'] <= lons[1]) & 
                          (ds['lat'] >= lats[0]) & (ds['lat'] <= lats[1]), drop=True)
            
        return ds
        
    def get_track_points(self, ds, var='PSL', bbox='southeast', new_unit=None):
        """
        Function to subset xarray DataSet spatially and by
        specific targeted variable. Returns xr.Dataset for
        native data and xr.DataArray for regridded data.
        Ensures that if variable passed in is not minimum sea level
        pressure (i.e. wind), that the coordinates ARE for min SLP.
        
        Returns
        ----------------
        data :: xr.Dataset or xr.DataArray
        """

        # Subsets data spatially
        ds = self.spatial_subset(ds, bbox)

        # Subsets data temporally into 6-hourly
        ds = ds.sel(time=ds.time.dt.hour.isin([0, 6, 12, 18]))

        # Converts units in place, if requested
        if new_unit:
            ds[var] = ds[var].metpy.convert_units(new_unit)
            ds[var] = ds[var].metpy.dequantify()

        # If the dataset is regridded and has lat/lon coords
        if 'lon' in ds.dims:
            if var == 'PSL':
                data = ds[var].isel(ds[var].compute().argmin(dim=('lon', 'lat')))
            else:
                values = ds[var].max(dim=('lon', 'lat'))
                loc_values = ds['PSL'].compute().argmin(dim=('lon', 'lat'))
                loc_da = ds[['lon', 'lat']].isel(loc_values)
                data = xr.DataArray(values, coords={'lon':loc_da['lon'], 'lat':loc_da['lat']})

        # If the dataset is native and doesn't have lat/lon coords
        else:
            if var == 'PSL':
                values = ds[var].compute().argmin(dim='ncol')
                data = ds[[var, 'lon', 'lat']].sel(ncol=values)
            else:
                values = ds[var].compute().argmax(dim='ncol')
                data_da = ds[var].sel(ncol=values)

                loc_values = ds['PSL'].compute().argmin(dim='ncol')
                loc_da = ds[['lon', 'lat']].sel(ncol=loc_values)

                loc_da[var] = data_da
                data = loc_da

        return data
    
    def check_err_pts(self, track):
        """
        Function to check if model spin-up is present.
        Returns track without erroneous points (for plotting).
        
        Parameters
        -------------
        track :: (xr.Dataset, xr.DataArray, or pd.DataFrame), 
                    Storm track with coordinates.
        """
        
        # Raises error if incorrect variable type
        if isinstance(track, (xr.Dataset, xr.DataArray, pd.DataFrame)):
            lons = track.lon.values
            lats = track.lat.values
        else:
            msg = 'Must supply an xarray Dataset/DataArray or pandas DataFrame.'
            raise TypeError(msg)

        # Converts longitudes from [0, 360] to [-180, 180] for haversine
        lons = np.mod(lons - 180, 360) - 180

        # Breaks coordinates into paired numpy arrays
        coords1 = np.stack([lons, lats], 1)

        # Shifts coordinates down once for haversine calculation
        coords2 = np.roll(coords1, 1, axis=0)

        # Calculates distance between points
        dists =  hv(coords1, coords2, unit='km')[1:]

        # Finds first index where distance > 350 km
        try:
            problem_idx1 = np.where(dists > 350)[0][0]

            # Denotes midpoint of data
            midpoint = int(len(lons)/2)

            # If the problem index is < midpoint, returns data after that
            if problem_idx < midpoint:
                track = track.isel(time=slice(problem_idx+1, None))

            # If the problem index > midpoint, returns data before that
            elif problem_idx > midpoint:
                track = track.isel(time=slice(None, problem_idx))
                
            return track
        
        except IndexError:
            return track
    
    def get_ts_metrics(self, data, storm_name, bbox, regridded=False):
        """
        Function to retrieve a storm's minimum sea level pressure
        and maximum surface wind speed values. Also converts SLP
        to hPa and WSP to mph. Returns list of DataArrays.
        
        Parameters
        --------------
        data       :: dict, output from ModelData
        storm_name :: str, name of target storm
        bbox       :: str or dict, feed to basin_bboxes
        regridded  :: bool, default=False, whether or not
                        to retrieve values for regridded data.
        """

        if regridded == True:
            ds_arr = ['h3pn_ds', 'h3cn_ds', 'h3pr_ds', 'h3cr_ds']
        elif regridded == False:
            ds_arr = ['h3pn_ds', 'h3cn_ds']

        # Packs target datasets into list
        ds_list = list(map(lambda x: data[storm_name][x], ds_arr))

        # Subsets data spatially
        subsets = list(map(self.spatial_subset, ds_list, [bbox]*len(ds_arr)))

        # Finds minimum SLPs and max WSPs, packs into lists
        try:
            min_slps = list(map(lambda x: x.PSL.min(dim='ncol'), subsets))
            max_wsps = list(map(lambda x: x.U10.max(dim='ncol'), subsets))
        except:
            min_slps = list(map(lambda x: x.PSL.min(dim=('lon', 'lat')), subsets))
            max_wsps = list(map(lambda x: x.U10.min(dim=('lon', 'lat')), subsets))

        # Converts MSLP and WSP to mb and mph, respectively.
        min_slps = list(map(lambda x: (x * units('Pa')).metpy.convert_units('hPa'), max_wsps))
        max_wsps = list(map(lambda x: (x * units('m/s')).metpy.convert_units('mph'), max_wsps))

        return min_slps, max_wsps
        
    def get_subset_idx(self, ds, bbox, overland=True, land_ds=None):
        """
        Retrieves the grid cell indices of a selected bounding
        box for easier unstructured grid plotting. 
        
        Parameters
        -------------
        overland :: bool, default=True; selects grid cells within
                    the defined bounding box, but only if they are
                    over land.
        
        Returns
        -------------
        idx :: np.ndarray of (integer) indices
        """
        
        # Retrieves bounding box coordinates
        if isinstance(bbox, (dict, str)):
            lons, lats = basin_bboxes(bbox)
        else:
            lons, lats = bbox
    
        # Gets indices to feed to ncol (for plotting)
        if 'ncol' in ds.dims:
            # Retrieves subset but leaves nans in place
            sub_ds = ds.where((ds['lon'] >= lons[0]) & (ds['lon'] <= lons[1]) &
                              (ds['lat'] >= lats[0]) & (ds['lat'] <= lats[1]))
    
            # Retrieves latitude and longitude values
            sub_lons = sub_ds.lon.values
            sub_lats = sub_ds.lat.values
    
            # Gets ncol indices of non-nan values
            lon_idx = np.argwhere(~np.isnan(sub_lons)).flatten()
            lat_idx = np.argwhere(~np.isnan(sub_lats)).flatten()
    
            # Gets union of both indices
            sub_idx = np.intersect1d(lon_idx, lat_idx)
    
            if overland == True:
                if not isinstance(land_ds, xr.Dataset):
                    raise ValueError('Must supply `land_ds`.')
                
                # Masks out land values
                land_idx = self.mask_land(ds, land_ds, True)[1]
        
                # Also gets union of subset indices and landmask indices
                idx = np.intersect1d(sub_idx, land_idx)
            else:
                idx = sub_idx
            
            return idx
        else:
            raise ValueError("'ncol' not in dataset dimensions.")
            
    def mask_land(self, ds, land_ds, return_idx=False):
        """
        Function to filter dataset by cells only located over land.
        """
        
        if isinstance(land_ds, xr.Dataset):
            # Retrieves grid cell indices over land
            idx = np.where(land_ds['LANDFRAC'] >= 0.5)[0]
        else:
            raise TypeError('Must supply xr.Dataset to `land_ds`.')
            
        # Subsets dataset by land indices
        ds = ds.isel(ncol=idx)
        
        # Returns indices if requested
        if return_idx == True:
            return ds, idx
        else:
            return ds
            

def geog_features(ax, basin='florida', proj=ccrs.PlateCarree(), res='10m', features={'plot all':True}, **kwargs):
    """
    Function to set up geographic features from cartopy.

    Parameters
    -------------------
    ax    :: mpl.axes
    basin :: str, optional - target basin, fed into uviz.plotting.utils.basin_bboxes
    proj  :: ccrs, optional - default is PlateCarree
    res   :: str, optional - default is '10m', resolution of geographic features.
                                Options are '10m', '50m', or '110m'.

    Optional kwargs
    -------------------
    features  :: dict, boolean - options are: 'coastline', 'states', 'lakes', and/or 'ocean';
                                    Set each one to True or False. Default == 'plot all'.
    coastline_kw :: dict - coastline plotting options
    states_kw    :: dict - states plotting options
    lakes_kw     :: dict - lakes plotting options
    ocean_kw     :: dict - ocean plotting options
    """

    # Ensures features variable is supplied correctly
    if not isinstance(features, dict):
        # Raises error if `features` isn't supplied as a dict
        raise TypeError("`features` must be supplied as a dict.")
    else:
        acceptable_vars = ['coastline', 'states', 'lakes', 'ocean', 'plot all']
        # Raises error if keys supplied to features aren't acceptable options
        if not set(features).issubset(set(acceptable_vars)):
            raise KeyError(f"Options for `features` dict are: {acceptable_vars}.")
        else:
            if not all(isinstance(value, bool) for value in features.values()):
                # Raises error if values supplied to features aren't booleans.
                raise ValueError("Not all values in `features` are boolean.")

    # If kwargs aren't supplied, set defaults
    coastline_kw = kwargs.get('coastline_kw', dict(linewidth=0.5, edgecolor='#323232', zorder=1.3))
    states_kw = kwargs.get('states_kw', dict(linewidth=0.5, facecolor='#EBEBEB', edgecolor='#616161', zorder=1.2))
    lakes_kw = kwargs.get('lakes_kw', dict(linewidth=0.5, facecolor='#e4f1fa', edgecolor='#616161', zorder=1.2))
    ocean_kw = kwargs.get('ocean_kw', dict(facecolor='#e4f1fa', edgecolor='face', zorder=1.1))

    # Sets up plotting spatial bounds
    if basin != None:
        lons, lats = basin_bboxes(basin)
        ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=proj)

    # Plots geographic features
    if 'plot all' in features.keys():
        ax.add_feature(cfeature.COASTLINE.with_scale(res), **coastline_kw)
        ax.add_feature(cfeature.STATES.with_scale(res), **states_kw)
        ax.add_feature(cfeature.LAKES.with_scale(res), **lakes_kw)
        ax.add_feature(cfeature.OCEAN.with_scale(res), **ocean_kw)
    else:
        if 'coastline' in features.keys():
            ax.add_feature(cfeature.COASTLINE.with_scale(res), **coastline_kw)
        if 'states' in features.keys():
            ax.add_feature(cfeature.STATES.with_scale(res), **states_kw)
        if 'lakes' in features.keys():
            ax.add_feature(cfeature.LAKES.with_scale(res), **lakes_kw)
        if 'ocean' in features.keys():
            ax.add_feature(cfeature.OCEAN.with_scale(res), **ocean_kw)

class CustomColorbars():
    """
    Class to store my custom outgoing longwave radiation and recipitation colormaps.
    """
    def __init__(self, target_cmap):
        
        self.out_cmap, self.out_levels, self.norm = self.get_cmap(target_cmap)
        
    def get_cmap(self, target_cmap):
        
        if target_cmap == 'FLUT':
            brightness_temps = np.array([-110, -92.1, -92, -80, -70, -60, -50, -42, 
                                         -30, -29.9, -20, -10, 0, 10, 20, 30, 40, 57])
            
            levels = np.array(list(map(self.T_to_FLUT, brightness_temps)))
            fracs = levels-self.T_to_FLUT(brightness_temps[0], 'C')
            fracs = fracs/fracs[-1]

            flut_colors = ['#ffffff', '#ffffff', '#e6e6e6', '#000000', 
                           '#ff1a00', '#e6ff01', '#00e30e', '#010073', 
                           '#00ffff', '#bebebe', '#acacac', '#999999', 
                           '#7e7e7e', '#6c6c6c', '#525252', '#404040', 
                           '#262626', '#070707']
            
            colormap = mcolors.LinearSegmentedColormap.from_list('FLUT CIMSS', list(zip(fracs, flut_colors)), N=1200)
            norm = mcolors.Normalize(vmin=levels[0], vmax=levels[-1])
        
        elif target_cmap == 'nws_precip':
            nws_precip_colors = [
                "#ffffff",
                "#ffffff",  # 0.00 - 0.10 inches  white        [0]
                "#4bd2f7",  # 0.10 - 0.25 inches  light blue   [1]
                "#699fd0",  # 0.25 - 0.50 inches  mid blue     [2]
                "#3c4bac",  # 0.50 - 1.00 inches  dark blue    [3]
                "#3cf74b",  # 1.00 - 2.00 inches  light green  [4]
                "#3cb447",  # 2.00 - 3.00 inches  mid green    [5]
                "#3c8743",  # 3.00 - 4.00 inches  dark green   [6]
                "#1f4723",  # 4.00 - 5.00 inches  darkest green[7]
                "#f7f73c",  # 5.00 - 6.00 inches  yellow       [8]
                "#fbde88",  # 6.00 - 8.00 inches  weird tan    [9]
                "#f7ac3c",  # 8.00 - 10.00 inches  orange      [10]
                "#c47908",  # 10.00 - 15.00 inches  dark orange[11]
                "#f73c3c",  # 15.00 - 20.00 inch  bright red   [12]
                "#bf3c3c",  # 20.00 - 25.00 inch  mid red      [13]
                "#6e2b2b",  # 25.00 - 30.00 inch  dark red     [14]
                "#f73cf7",  # 30.00 - 35.00 inch  bright pink  [15]
                "#9974e4",  # 40.00 - 45.00 inch  purple       [16]
                #"#404040",  # 30.00 - 40.00 inch  dark gray because of mpl
                "#c2c2c2"  # 45.00 - 50.00 inch  gray          [17]
                ]
            colormap = mcolors.ListedColormap(nws_precip_colors, 'nws_precip')
            levels = [0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 
                      10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
            #levels = [0.01, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 
            #4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(levels))
            
        elif target_cmap == 'shear_cimss':
            colors = ["#00ff00",    # green (for 0-15 kts)
                      "#fffe00",    # yellow (for 20 kts)
                      *[*["#ff0200"]*8]]    # red (for >20 kts)
            colormap = mcolors.ListedColormap(colors, 'shear_cimss')
            levels = [0.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(levels))

        else:
            raise ValueError("Must choose either 'FLUT', 'nws_precip', or 'shear_cimss' colormap.")
        
        return colormap, levels, norm

    def T_to_FLUT(self, T, unit='C'):
        """
        Converts brightness temperature to outgoing longwave 
        radiation flux using the Stefan-Boltzmann Law. 
        Used to make pseudo-enhanced satellite plots.
        
        Parameters:
        ----------------
        T    :: float, temperature
        unit :: str (optional), unit of temperature. Default is 'C'.
        """
        
        if unit == 'C':
            T += 273.15
        sigma = 5.6693E-8
        olr = sigma*(T**4)

        return olr
    
def diverging_colormap(cmin, cmid, cmax, palette, ncolors=256):
    """
    Function to create a diverging colormap to use with Holoviews (which is not natively supported).
    Returns divergent colormap (one where the normalization above the midpoint != normalization 
    below the midpoint).
    
    Parameters:
    ----------------
    cmin     ::  float, int - minimum to clip plotting data (akin to mpl's vmin) for normalization
    cmid     ::  float, int - midpoint to begin divergence of colors
    cmax     ::  float, int - maximum to clip plotting data (akin to mpl's vmax) for normalization
    palette  ::  bokeh.palettes, list - original colormap (or list of colors) for diverging
    
    ncolors  :: optional, int (default=256), must match number of colors in the supplied palette.
    """

    diverge_point_norm = (cmid - cmin) / (cmax - cmin)
    palette_cutoff = round(diverge_point_norm * ncolors)
    palette_split = palette[palette_cutoff:]
    diverge_cmap = bokeh.palettes.diverging_palette(palette[:palette_cutoff], palette_split[::-1], n=ncolors, midpoint=diverge_point_norm)
    
    return diverge_cmap

class SaffirSimpson():
    def __init__(self, var, units):
        self.units = units
        self.var = var
        wsp_units = ['m/s', 'knots', 'kts', 'mph', 'miles_per_hour', 'mile_per_hour']
        mslp_units = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals', 
                      'mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']
        
        if self.units in wsp_units:
            self.category = self.sshs_wsp(self.var, self.units)
        elif self.units in mslp_units:
            self.category = self.sshs_mslp(self.var, self.units)
        else:
            raise ValueError(f'Supplied units not recognized: {self.units}')
            
        self.color = self.sshs_color()
        self.lw = self.sshs_lw()
        
    def sshs_wsp(self, wsp, units):
        """
        Takes wind speed, converts to mph, returns corresponding category from SSHWS
        Parameters:
        -------------
        wsp : ndarray or variable
        units : str
        
        Returns: category : ndarray
        """
        # Converts units to mph so I only needed to write one function.
        if units == 'm/s':
            wsp = wsp * 2.2369
        elif units in ['knots','kts']:
            wsp = wsp * 1.15077945
        elif units in ['mph', 'miles_per_hour', 'mile_per_hour']:
            wsp = wsp
        else:
            raise ValueError("Wind speed units must be 'm/s', 'kts', or 'mph'.")
            
        # Categorizes wind speed
        category = np.vectorize(lambda x: 'Category 5' if x >= 157.0 
                                else ('Category 4' if np.logical_and(x < 157.0, x > 129.0) 
                                else ('Category 3' if np.logical_and(x > 110.0, x <= 129.0) 
                                else ('Category 2' if np.logical_and(x > 95.0, x <= 110.0) 
                                else ('Category 1' if np.logical_and(x >= 74.0, x <= 95.0) 
                                else ('Tropical Storm' if np.logical_and(x > 38.0, x < 74.0) 
                                else 'Tropical Depression'))))))(wsp)
        
        return category
        

    def sshs_mslp(self, slp, unit):
        """
        Takes minimum sea level pressure, converts to pascals, returns corresponding category from SSHS.
        (According to Klotzbach et. al., 2020, modified for TS and TD designation using Kantha, 2006)
        Parameters:
        -------------
        wsp : ndarray or variable
        units : str
        """
        pascals = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals']
        hPa = ['mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']

        if unit in pascals:
            slp = slp/100

        category = np.vectorize(lambda x: 'Category 5' if x <= 925.0 
                                else ('Category 4' if np.logical_and(x <= 946.0, x > 925.0) 
                                else ('Category 3' if np.logical_and(x <= 960.0, x > 946.0) 
                                else ('Category 2' if np.logical_and(x <= 975.0, x > 960.0) 
                                else ('Category 1' if np.logical_and(x <= 990.0, x > 975.0) 
                                else ('Tropical Storm' if np.logical_and(x < 1005.0, x > 990.0) 
                                else 'Tropical Depression'))))))(slp)
        return category

    def sshs_color(self):
        """
        Takes category on the Saffir Simpson Hurricane Scale, spits out correct color for plotting.
        """
        color = np.vectorize(lambda x: '#5EBAFF' if x == 'Tropical Depression'
                             else ('#00FAF4' if x == 'Tropical Storm'
                            else ('#FFF795' if x == 'Category 1' 
                            else ('#FFD821' if x == 'Category 2' 
                            else ('#FF8F20' if x == 'Category 3' 
                            else ('#FF6060' if x == 'Category 4' 
                            else '#C464D9' if x == 'Category 5' 
                            else ''))))))(self.category)
        return color
    
    def sshs_lw(self):
        """
        Takes category on the Saffir Simpson Hurricane Scale, 
        spits out correct linewidth for plotting. Only for
        spaghetti plots to make them less messy.
        """
        lw = np.vectorize(lambda x: 0.1 if x == 'Tropical Depression'
                             else (0.2 if x == 'Tropical Storm'
                            else (0.3 if x == 'Category 1' 
                            else (0.4 if x == 'Category 2' 
                            else (0.5 if x == 'Category 3' 
                            else (0.6 if x == 'Category 4' 
                            else 0.7 if x == 'Category 5' 
                            else ''))))))(self.category)
        return lw
        
    
def basin_bboxes(basin_name, lon_range=360, **kwargs):
    """
    Creates predefined bounding boxes if supplied correct name.
    
    Parameters:
    -------------------
    basin_name :: str or dict, target basin name or dict of values. If dict,
                        options include: {'center_coords', 'distance', 'unit'}, or
                            {'west_coord', 'east_coord', 'north_coord', 'south_coord'}.
    lon_range  :: int (optional), defines maximum range of longitude. 
                        Options are 360 or 180, default is 360.
    """
    
    if isinstance(basin_name, str):
        basin_name = basin_name.lower()
        custom_basin = {}
    elif isinstance(basin_name, dict):
        custom_basin = basin_name
    else:
        raise KeyError('Must supply string or dict to basin_name')
        
    basins = ['north atlantic', 'north atlantic zoomed', 'florida', 'south florida', 
              'south atlantic', 'east pacific', 'west pacific', 'north indian', 'south indian', 
              'australia', 'south pacific', 'conus', 'east conus', 'southeast']
    
    if basin_name not in basins and not custom_basin:
        err = f"'{basin_name}' not in list of basins. Choose from {basins} or supply dict."
        raise ValueError(err)
        
    # Sets defaults primarily for DDF curves
    if 'basin_name' in custom_basin.keys():
        if custom_basin['basin_name'] == 'miami':
            custom_basin['center_coords'] = (25.775163, -80.208615)
            if 'distance' not in custom_basin.keys():
                custom_basin['distance'] = 100
            if 'unit' not in custom_basin.keys():
                custom_basin['unit'] = 'km'
        
    # Option 1: supply a name for one of the predetermined bounding boxes.
    if basin_name == 'north atlantic':
        west_coord = -105.0
        east_coord = -5.0
        north_coord = 70.0
        south_coord = 0.0
    elif basin_name == 'north atlantic zoomed':
        west_coord = -100.0
        east_coord = -15.0
        north_coord = 50.0
        south_coord = 7.5
    elif basin_name == 'florida':
        west_coord = -90.0
        east_coord = -72.5
        north_coord = 32.5
        south_coord = 20.0
    elif basin_name == 'southeast':
        west_coord = -94.0
        east_coord = -70.0
        north_coord = 36.0
        south_coord = 20.0
    elif basin_name == 'south florida':
        west_coord = -84.0
        east_coord = -79.0
        north_coord = 30.0
        south_coord = 24.0
    elif basin_name == 'south atlantic':
        west_coord = -105.0
        east_coord = -5.0
        north_coord = 65.0
        south_coord = 0.0
    elif basin_name == 'east pacific':
        west_coord = -180.0
        east_coord = -80.0
        north_coord = 65.0
        south_coord = 0.0    
    elif basin_name == 'west pacific':
        west_coord = 90.0
        east_coord = 180.0
        north_coord = 58.0
        south_coord = 0.0        
    elif basin_name == 'north indian':
        west_coord = 35.0
        east_coord = 110.0
        north_coord = 42.0
        south_coord = -5.0    
    elif basin_name == 'south indian':
        west_coord = 20.0
        east_coord = 110.0
        north_coord = 5.0
        south_coord = -50.0        
    elif basin_name == 'australia':
        west_coord = 90.0
        east_coord = 180.0
        north_coord = 0.0
        south_coord = -60.0        
    elif basin_name == 'south pacific':
        west_coord = 140.0
        east_coord = -120.0#+360
        north_coord = 0.0
        south_coord = -65.0        
    elif basin_name == 'conus':
        west_coord = -130.0
        east_coord = -65.0
        north_coord = 50.0
        south_coord = 20.0        
    elif basin_name == 'east conus':
        west_coord = -105.0
        east_coord = -60.0
        north_coord = 48.0
        south_coord = 20.0
    elif custom_basin:
        set1 = {'center_coords', 'distance', 'unit'}
        set2 = {'west_coord', 'east_coord', 'north_coord', 'south_coord'}
        
        # Option 2: supply a center point and a distance around it to return a bounding box.
        if set1 <= custom_basin.keys():
            center_pt = custom_basin['center_coords']
            dist = custom_basin['distance']
            acceptable_units = [item.value for item in Unit]
            
            if custom_basin['unit'] not in acceptable_units:
                raise ValueError(f"Acceptable units are: {acceptable_units}.")
            else:
                unit = custom_basin['unit']
                
            west_coord = inverse_haversine(center_pt, dist, Direction.WEST, unit=unit)[1]
            east_coord = inverse_haversine(center_pt, dist, Direction.EAST, unit=unit)[1]
            north_coord = inverse_haversine(center_pt, dist, Direction.NORTH, unit=unit)[0]
            south_coord = inverse_haversine(center_pt, dist, Direction.SOUTH, unit=unit)[0]
            
        # Option 2: supply a user-defined bounding box.
        elif set2 <= custom_basin.keys():
            west_coord = custom_basin['west_coord']
            east_coord = custom_basin['east_coord']
            north_coord = custom_basin['north_coord']
            south_coord = custom_basin['south_coord']
        
        # There are no other custom options. (Fix this later)
        else:
            err = f"Must supply either all of {str(set1)} or all of {str(set2)} in `custom_basin` keys."
            raise KeyError(err)

    # There are no other options.
    else:
        raise ValueError("Supplied name not in given list.")
    
    # Converts longitude from [-180, 180] to [0, 360].
    if lon_range == 360:
        west_coord = np.mod(west_coord, 360)
        east_coord = np.mod(east_coord, 360)
        
    # Currently doesn't support radians.
    elif lon_range != 360 and lon_range != 180:
        raise ValueError("Acceptable values for `lon_range` are 180 or 360.")
        
    # Packs longitude range and latitude range into lists and returns them.
    lons = (west_coord, east_coord)
    lats = (south_coord, north_coord)
    
    return lons, lats
