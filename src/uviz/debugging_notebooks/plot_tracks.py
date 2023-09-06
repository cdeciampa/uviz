import os, glob
import xarray as xr

import metpy.calc as mpcalc
from metpy.units import units

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from haversine import haversine
from haversine import haversine_vector as hv

#from uviz.plotting.processing import ModelData
#from uviz.plotting.utils import SaffirSimpson, basin_bboxes
#from uviz.tempest_extremes.processing import TempestExtremes

class TempestExtremes():
    """
    Class object that reads in TempestExtremes file (ASCII or .cvs), 
    cleans up data, and assigns to pandas dataframe.
    """
    def __init__(self, file, colnames=None):
        
        self.file = file
        self.colnames = colnames
        
        if '.csv' in self.file:
            self.df = self.read_te_csv(self.file)
        else:
            self.df = self.read_te_ASCII(self.file)
            
        self.track_IDs = self.df['tempest_ID'].unique().tolist()
        
    def read_te_ASCII(self, file):
        """
        Function that reads in TempestExtremes ASCII text files
        and assigns to pandas dataframe. Also cleans up date for
        easier datetime parsing in pandas and assigns new column
        to categorize and name individual tracks.

        https://github.com/ClimateGlobalChange/tempestextremes

        Parameters
        --------------
        file     :: path to ASCII text file
        colnames :: optional, list - names to use for columns.

        Returns
        --------------
        pandas DataFrame
        """

        # Declares column names (defaults of TE) if none supplied
        if self.colnames == None:
            colnames = ['i', 'j', 'lon', 'lat', 'slp', 'wind', 'phi', 
                        'year', 'month', 'day', 'hour']
            # Parent track file column headers are a little different
            if 'NATL' in self.file:
                colnames = ['timestep', 'lon', 'lat', 'slp', 'wind', 
                            'phi', 'year', 'month', 'day', 'hour']
        elif not isinstance(colnames, (list, np.ndarray)):
            raise ValueError('Must supply a list of column names.')

        # Reads in file
        try:
            df = pd.read_csv(file, sep='\s+', names=colnames)
        except pd.errors.ParserError as err:
            msg = f'Number of supplied column names != number of columns in file{str(err)[30:]}'
            raise ValueError(msg)

        # Identifies individual year, month, day, hour columns
        datetime_cols = ['year', 'month', 'day', 'hour']

        # Takes individual year, month, day, hour columns and transforms to pd.date_time column
        df['time'] = pd.to_datetime(df[datetime_cols], errors='coerce')

        # Drops year, month, day, hour columns
        df = df.drop(datetime_cols, axis=1)

        # Selects indices where a new track starts, assigns to array
        run_idx = df[df[df.columns[0]]=='start'].index.tolist()

        # Separates dataframe into individual dataframes, split on new tracks
        dfs = [df.iloc[run_idx[n]+1:run_idx[n+1]] for n in range(len(run_idx)-1)]

        # Only keeps selected columns
        target_cols = ['time', 'lon', 'lat', 'slp', 'wind', 'phi']
        dfs = [dfi[target_cols] for dfi in dfs]

        # Resets index from previous dataframe splits
        dfs = [dfi.reset_index(drop=True) for dfi in dfs]

        # Creates storm IDs for each track (IDs reset at 0 for each year)
        for i, dfi in enumerate(dfs):
            dfi['tempest_ID'] = f'storm_{str(i).zfill(4)}'

        # Merges dfs back together
        df_concat = pd.concat([dfi for dfi in dfs]).reset_index(drop=True)

        return df_concat

    def read_te_csv(self, file):
        """
        Function to read in TempestExtremes csv files. This is
        specific to my research and is a non-generalized function.

        Parameters
        ---------------
        file :: .csv, headers = [track_id, year, month, day, hour, 
                                    i, j, lon, lat, slp, wind, phis]

        Returns
        ---------------
        pandas DataFrame
        """

        # Reads in file, swaps outrageous wind values with NaN
        df = pd.read_csv(file, na_values='14142.00000')

        # Strips leading space in most columns
        df = df.rename(columns=lambda x: x.strip())

        # Takes individual year, month, day, hour columns and transforms to pd.date_time column
        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')

        # Drops year, month, day, hour, i, j, and phis columns
        df = df[['track_id', 'time', 'lon', 'lat', 'slp', 'wind']]

        # Adds 'storm' as prefix to track_id
        df['track_id'] = 'storm_' + df['track_id'].apply(lambda x: str(x).zfill(4))

        # Renames track_id column for processing with ibtracs
        df = df.rename(columns={'track_id':'tempest_ID'})

        return df

class SaffirSimpson():
    def __init__(self, var, units):
        self.units = units
        self.var = var
        wsp_units = ['m/s', 'knots', 'kts', 'mph', 'miles_per_hour']
        mslp_units = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals', 
                      'mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']
        
        if self.units in wsp_units:
            self.category = self.sshs_wsp(self.var, self.units)
        elif self.units in mslp_units:
            self.category = self.sshs_mslp(self.var, self.units)
        else:
            raise ValueError(f'Supplied units not recognized: {self.units}')
            
        self.color = self.sshs_color()
        
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
        elif units == 'knots' or units == 'kts':
            wsp = wsp * 1.15077945
        elif units == 'mph' or units =='miles_per_hour':
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
                                else ('Tropical Storm' if np.logical_and(x < 1000.0, x > 990.0) 
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
              'australia', 'south pacific', 'conus', 'east conus']
    
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
        
        # There are no other custom options.
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

class TrackPlot():
    
    def __init__(self, storm_dict=None, savefig=False):
        
        self.data = storm_dict
        self.savefig = savefig
        
    def add_legend(self, ax, feat_type='points', **kwargs):
    
        #plot_kw = kwargs.get('plot_kw', dict(ls))

        # Retrieves SSHWS labels and hex colors
        cat_mags = [160, 135, 115, 100, 75, 50, 35]
        cats = SaffirSimpson(units='mph', var=cat_mags)
        cat_labels = cats.category
        cat_colors = cats.color

        if feat_type == 'segments':
            lw = 1.5
            style_dict = {'xdata':[], 'ydata':[], 'ls':'-', 'lw':lw, 
                          'path_effects':[pe.Stroke(linewidth=lw*1.5, foreground='#848484'), pe.Normal()]}
            cats_handles = [mlines.Line2D(label=l, color=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]
            legend_kw = {'handles':cats_handles, 'loc':'upper right', 'shadow':True, 'fontsize':6, 'handlelength':1.0}

        elif feat_type == 'points':   
            # Marker properties
            mew = 0.35      # marker edge width
            mec = 'k'      # marker edge color
            ms = 5         # marker size
            #mfc           # marker face color

            style_dict = {'xdata':[], 'ydata':[], 'marker': 'o', 'ms':ms, 
                          'mew': mew, 'color':'k', 'ls':'-', 'lw':0.5}
            cats_handles = [mlines.Line2D(label=l, mfc=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]
            legend_kw = {'handles':cats_handles, 'loc':'upper right', 'shadow':True, 'fontsize':8}

        l = ax.legend(**legend_kw)
        l.set_zorder(1001)

    def plot_track(self, ax, data, bbox, feat_type='points', proj=ccrs.PlateCarree(), unit='pressure', legend=True, **kwargs):
        """
        Function to make track plots categorized by the Saffir-Simpson 
        Hurricane Scale. Can plot either color-coded segments alone or 
        plain black segments with points on top. Returns mpl plot.

        Parameters
        -------------------
        ax        :: mpl.axes - axes to plot on
        data      :: xr.Dataset, xr.DataArray, pd.DataFrame - data to be plotted

        Optional Params
        -------------------
        feat_type :: str - options are 'points' or 'segments', default='points'
        proj      :: ccrs - cartopy projection type, default=ccrs.PlateCarree()
        unit      :: str - options are 'pressure' or 'wind', default='pressure'
        legend    :: bool - whether or not to plot the legend, default=True

        Optional kwargs
        -------------------
        figtitle    :: str, default=None - text of figure title
        points_kw   :: dict - point style options to pass to mpl
        segments_kw :: dict - line style options to pass to mpl
        title_kw    :: dict - title font style options to pass to mpl
        geog_kw     :: dict - kwargs to pass to geog_features
        """

        # If kwargs aren't supplied, set defaults
        figtitle = kwargs.get('figtitle', None)
        points_kw = kwargs.get('point_kw', {})
        segments_kw = kwargs.get('segments_kw', {})
        title_kw = kwargs.get('title_kw', {})
        geog_kw = kwargs.get('geog_kw', {})
        
        # Plots geographic features
        self.geog_features(ax, bbox, **geog_kw)

        # Sets defaults for plotting variables
        if unit == 'pressure':
            te_var = 'slp'
            te_unit = 'Pa'
        elif unit == 'wind':
            te_var = 'wind'
            te_unit = 'm/s'

        # For use with TempestExtremes dataframe
        if isinstance(data, pd.DataFrame):

            for track, track_df in data.groupby('tempest_ID'):

                # Pulls coordinate values
                lons = track_df['lon'].values
                lats = track_df['lat'].values

                # Categorizes data based on SSHS
                sshws_cmap = SaffirSimpson(track_df[te_var], units=te_unit).color

                # Packs data into points and segments connecting points
                points = np.array([lons, lats]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                if feat_type == 'points':
                    # Updates kwargs in place, prioritizing supplied kwargs over defaults
                    points_style = {'edgecolors':'k', 'lw':0.35, 's':17.5, 'transform':proj, 'zorder':10}
                    points_kw = {**points_style, **points_kw}

                    # Plots track points
                    ax.scatter(lons, lats, c=sshws_cmap, **points_kw)

                    # Updates kwargs in place, prioritizing supplied kwargs over defaults
                    segments_style = {'zorder': 9, 'transform':proj, 'lw':0.5, 'ls':'-', 'colors':'k'}
                    segments_kw = {**segments_style, **segments_kw}

                elif feat_type == 'segments':
                    # Updates kwargs in place, prioritizing supplied kwargs over defaults
                    lw = 0.35
                    segments_style = {'zorder':10, 'transform':proj, 'lw':lw, 'colors': sshws_cmap, 
                                      'path_effects': [pe.Stroke(linewidth=lw*2, foreground='#848484'), 
                                                       pe.Normal()]}
                    segments_kw = {**segments_style, **segments_kw}

                else:
                    raise ValueError("Must choose 'points' or 'segments' for `feat_type`.")

                # Plots segments between points
                line_segments = LineCollection(segments, **segments_kw)
                ax.add_collection(line_segments)

        # For use with parent and child xarray datasets
        elif isinstance(data, (xr.Dataset, xr.DataArray)):

            # Slices data by 6-hourly segments
            data = data.sel(time=data.time.dt.hour.isin([0, 6, 12, 18]))

            # Checks if there are erroneous track points
            data = self.check_err_pts(data)

            # Checks again in case erroneous points bookend track
            data = self.check_err_pts(data)

            # Pulls coordinate values
            lons = data.lon.values
            lats = data.lat.values

            # Categorizes data based on SSHS
            if isinstance(data, xr.Dataset):
                var = list(child_mslps.keys())[0]
                units = data[var].units
                sshws_cmap = SaffirSimpson(data[var].values, units=units).color
            elif isinstance(data, xr.DataArray):
                sshws_cmap = SaffirSimpson(data.values, units=data.units).color

            # Packs data into points and segments connecting points
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if feat_type == 'points':
                # Updates kwargs in place, prioritizing supplied kwargs over defaults
                points_style = {'edgecolors':'k', 'lw':0.5, 's':75, 'transform':proj, 'zorder':10}
                points_kw = {**points_style, **points_kw}

                # Plots track points
                ax.scatter(lons, lats, c=sshws_cmap, **points_kw)

                # Updates kwargs in place, prioritizing supplied kwargs over defaults
                segments_style = {'zorder': 9, 'transform':proj, 'lw':0.5, 'ls':'-', 'colors':'k'}
                segments_kw = {**segments_style, **segments_kw}

            elif feat_type == 'segments':
                # Updates kwargs in place, prioritizing supplied kwargs over defaults
                segments_style = {'zorder':10, 'transform':proj, 'lw': 1.0, 'colors': sshws_cmap, 
                                  'path_effects': [pe.Stroke(linewidth=2.0, foreground='#848484'), 
                                                   pe.Normal()]}
                segments_kw = {**segments_style, **segments_kw}

            else:
                raise ValueError("Must choose 'points' or 'segments' for `feat_type`.")

            # Plots segments between points
            line_segments = LineCollection(segments, **segments_kw)
            ax.add_collection(line_segments)

        # Plots legend if True
        if legend == True:
            self.add_legend(ax, feat_type)

        if figtitle:
            # Updates kwargs in place, prioritizing supplied kwargs over defaults
            title_style = {'fontsize':10, 'fontstyle':'bold'}
            title_kw = {**title_style, **title_kw}
            plt.title(figtitle, **title_kw)
            
        #plt.show()
        #plt.close()
        # if self.savefig == True:
        #     plt.show()
        # else:
        #     plt.show()
            
    def check_err_pts(self, track):
        """
        Function to check if model spin-up is present.
        """

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
    
    def geog_features(self, ax, bbox='florida', proj=ccrs.PlateCarree(), res='10m', features={'plot all':True}, **kwargs):
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
        if bbox != None:
            lons, lats = basin_bboxes(bbox)
            ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=proj)

        # Plots geographic features
        if 'plot all' in features.keys():
            ax.add_feature(cfeature.COASTLINE.with_scale(res), **coastline_kw)
            ax.add_feature(cfeature.STATES.with_scale(res), **states_kw)
            if bbox == 'florida':
                ax.add_feature(cfeature.LAKES.with_scale(res), **lakes_kw)
            if bbox in ['north atlantic', 'north atlantic zoomed']:
                ocean_kw['facecolor'] = '#f2f8fd'
            ax.add_feature(cfeature.OCEAN.with_scale(res), **ocean_kw)
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


def miami_storms(df, distance=500.0, units='km'):

    miami_coords = (25.775163, -80.208615)
    miami_dist = distance #km
    df['miami_dist'] = df.apply(lambda x: haversine((x.lat, x.lon), miami_coords, unit=units, normalize=True), axis=1).round(2)
    df = df[df['miami_dist'].apply(lambda x: x <= miami_dist)].reset_index(drop=True)
    storms = df['tempest_ID'].unique()
    
    return storms


# Defines where trajectories files are
te_files = glob.glob('../tempest_extremes/trajectories.txt*VR28*')

# Reads trajectories files into a list of dataframes
te_dfs = [TempestExtremes(f).df for f in te_files]

# Identifies storms within 500km of miami
potential_tracks = [miami_storms(df) for df in te_dfs]

# Subsets original data by identified tracks
spaghetti_tracks = [df[df['tempest_ID'].isin(x)].reset_index(drop=True) for df, x in zip(te_dfs, potential_tracks)]

# Spaghetti track plot (for identifying tracks)
fig, axs = plt.subplots(3, 3, dpi=300, figsize=(8, 5), 
                        subplot_kw=dict(projection=ccrs.PlateCarree()), layout='constrained')

for i, ax in enumerate(axs.ravel()):
    if i == 2:
        legend = True
    else:
        legend = False
        
    TrackPlot().plot_track(ax, spaghetti_tracks[i], 'north atlantic zoomed', 'segments', legend=legend, unit='wind')
    ax.set_title(f'{sim_names[i]}', fontsize=9, fontweight='bold')
fig.suptitle('Potential Tracks Identified by TempestExtremes Within 500 km of Miami', fontweight='bold', fontsize=12)
plt.show()

# Track plots for 9 identified storms
sim_names = ['CHEY.EXT.001', 'CHEY.EXT.002', 'CHEY.REF.001', 
             'CHEY.REF.002', 'CHEY.WAT.001', 'CHEY.WAT.002', 
             'CORI.EXT.003', 'CORI.REF.003', 'CORI.WAT.003']
sim_names.sort()

selected_storms = ['storm_1279', '', 'storm_1048', 
                   ['storm_0236', 'storm_0528', 'storm_1521'], 'storm_0310', 'storm_1354', 
                   'storm_1307', 'storm_0755', '']
selected_tracks = [df[df['tempest_ID'].isin(x)].reset_index(drop=True) for df, x in zip(te_dfs, selected_storms)]

# Track plots (selected tracks)
fig, axs = plt.subplots(3, 3, dpi=300, figsize=(7, 6), 
                        subplot_kw=dict(projection=ccrs.PlateCarree()), layout='constrained')

for i, ax in enumerate(axs.ravel()):
    if i == 1:
        legend = True
    else:
        legend = False
        
    TrackPlot().plot_track(ax, selected_plots[i], 'florida', 'points', legend=legend, unit='pressure')
    ax.set_title(f'{sim_names[i]}', fontsize=9, fontweight='bold')
fig.suptitle('Selected Tracks for Downscaling', fontweight='bold', fontsize=13)
plt.show()
