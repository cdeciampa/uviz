import numpy as np
import pandas as pd

from haversine import haversine_vector as hv

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from uviz.plotting.utils import PlotProcessing, SaffirSimpson, basin_bboxes #, geog_features
from uviz.plotting.statistics import Statistics

class TrackPlot():
    
    def __init__(self, storm_dict=None, savefig=False):
        
        self.data = storm_dict
        self.savefig = savefig
        
    def add_legend(self, ax, feat_type='points', **kwargs):
        
        plot_kw = kwargs.get('plot_kw', {})

        # Retrieves SSHWS labels and hex colors
        cat_mags = [160, 135, 115, 100, 75, 50, 35]
        cats = SaffirSimpson(units='mph', var=cat_mags)
        cat_labels = cats.category
        cat_colors = cats.color

        if feat_type == 'segments':
            if not plot_kw:
                lw = 1.5
            style_dict = {'xdata':[], 'ydata':[], 'ls':'-', 'lw':lw, 
                          'path_effects':[pe.Stroke(linewidth=lw*1.5, foreground='#848484'), pe.Normal()]}
            style_kw = {**style_dict, **plot_kw}
            cats_handles = [mlines.Line2D(label=l, color=c, **style_kw) for (l, c) in zip(cat_labels, cat_colors)]
            legend_kw = {'handles':cats_handles, 'loc':'upper right', 'shadow':True, 'fontsize':6, 'handlelength':1.0}

        elif feat_type == 'points': 
            if not plot_kw:
                # Marker properties
                mew = 0.35      # marker edge width
                mec = 'k'      # marker edge color
                ms = 5         # marker size
                lw = 0.5       # segment line weight
                #mfc           # marker face color

            style_dict = {'xdata':[], 'ydata':[], 'marker': 'o', 'ms':ms, 
                          'mew': mew, 'color':'k', 'ls':'-', 'lw':lw}
            style_kw = {**style_dict, **plot_kw}
            cats_handles = [mlines.Line2D(label=l, mfc=c, **style_kw) for (l, c) in zip(cat_labels, cat_colors)]
            legend_kw = {'handles':cats_handles, 'loc':'upper right', 'shadow':True, 'fontsize':8}

        l = ax.legend(**legend_kw)
        l.set_zorder(1001)

    def plot_track(self, ax, data, bbox, feat_type='points', proj=ccrs.PlateCarree(), unit='pressure', legend=False, **kwargs):
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
        legend    :: bool - whether or not to plot the legend, default=False

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
        points_kw = kwargs.get('points_kw', {})
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
            
        if feat_type == 'points':
            # Updates kwargs in place, prioritizing supplied kwargs over defaults
            if isinstance(data, pd.DataFrame):
                point_size = 18 #17.5
                point_lw = 0.5 #0.35
            elif isinstance(data, (xr.Dataset, xr.DataArray)):
                point_size = 20
                point_lw = 0.5
            points_style = {'edgecolors':'k', 'lw':point_lw,
                            's':point_size, 'transform':proj, 'zorder':10}
            points_kw = {**points_style, **points_kw}
            segments_style = {'zorder': 9, 'transform':proj, 
                              'lw':0.8, 'ls':'-', 'colors':'k'} 
            segments_kw = {**segments_style, **segments_kw}
        elif feat_type == 'segments':
            # Updates kwargs in place, prioritizing supplied kwargs over defaults
            if isinstance(data, pd.DataFrame):
                lw = 0.4 #0.5 #0.35
                outline = '#818181'
            elif isinstance(data, (xr.Dataset, xr.DataArray)):
                lw = 1.0
                outline = '#818181'
            segments_style = {'zorder':10, 'transform':proj, 'lw':lw, 
                              'path_effects': [pe.Stroke(linewidth=lw*1.5, foreground=outline), 
                                               pe.Normal()]}
            segments_kw = {**segments_style, **segments_kw}
        else:
            raise ValueError("Must choose 'points' or 'segments' for `feat_type`.")
        
        # For use with IBTrACS and TempestExtremes dataframes
        if isinstance(data, pd.DataFrame):
            if 'tempest_ID' in data.columns:
                group_by = 'tempest_ID'
                lon='lon'
                lat='lat'
            elif 'SID' in data.columns:
                group_by = 'SID'
                lon='LON'
                lat='LAT'
                if unit == 'wind':
                    te_var = 'WSP'
                elif unit == 'pressure':
                    te_var = 'PRES'
            for track, track_df in data.groupby(group_by):

                # Pulls coordinate values
                lons = track_df[lon].values
                lats = track_df[lat].values

                # Categorizes data based on SSHS
                if te_var == 'wind' and group_by == 'tempest_ID':
                    track_df[te_var] = track_df[te_var]*0.85
                sshws_cmap = SaffirSimpson(track_df[te_var], units=te_unit).color

                # Packs data into points and segments connecting points
                points = np.array([lons, lats]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                if feat_type == 'points':
                    # Plots track points
                    ax.scatter(lons, lats, c=sshws_cmap, **points_kw)
                elif feat_type == 'segments':
                    sshws_lw = SaffirSimpson(track_df[te_var], units=te_unit).lw
                    try:
                        del segments_kw['lw']
                        #del segments_kw['path_effects']
                    except:
                        pass
                    
                    # for track_lw in sshws_lw:
                    #     strokes = [pe.Stroke(linewidth=track_lw*1.5, 
                    #                      foreground=outline), 
                    #            pe.Normal()]
                    #     segments_kw['path_effects'] = strokes
                    segments_kw['linewidths'] = sshws_lw
                    segments_kw['colors'] = sshws_cmap

                # Plots segments between points
                line_segments = LineCollection(segments, **segments_kw)
                ax.add_collection(line_segments)

        # For use with parent and child xarray datasets
        elif isinstance(data, (xr.Dataset, xr.DataArray)):

            # Slices data by 6-hourly segments
            data = data.sel(time=data.time.dt.hour.isin([0, 6, 12, 18]))

            # Checks if there are erroneous track points
            i=0
            while i < len(data.time):
                data = self.check_err_pts(data)
                i += 1

            # Checks again in case erroneous points bookend track
            #data = self.check_err_pts(data)

            # Pulls coordinate values
            lons = data.lon.values
            lats = data.lat.values

            # Categorizes data based on SSHS
            if isinstance(data, xr.Dataset):
                var = list(data.keys())[-1]
                if var in ['lon', 'lat']:
                    var = list(data.keys())[0]
                units = data[var].units
                sshws_cmap = SaffirSimpson(data[var].values, units=units).color
            elif isinstance(data, xr.DataArray):
                sshws_cmap = SaffirSimpson(data.values, units=data.units).color

            # Packs data into points and segments connecting points
            points = np.array([lons, lats]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            if feat_type == 'points':
                # Plots track points
                ax.scatter(lons, lats, c=sshws_cmap, **points_kw)
            elif feat_type == 'segments':
                segments_kw['colors'] = sshws_cmap

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
        coords1 = np.stack([lats, lons], 1)

        # Shifts coordinates down once for haversine calculation
        coords2 = np.roll(coords1, 1, axis=0)

        # Calculates distance between points
        dists =  hv(coords1, coords2, unit='km')[1:]

        try:
            # Finds first index where distance > 350 km
            problem_idx = np.where(dists > 350)[0][0]

            # Denotes midpoint of data
            midpoint = int(len(lons)/2)

            # If the problem index is < midpoint, returns data after that
            if problem_idx < midpoint:
                track = track.isel(time=slice(problem_idx+1, None))

            # If the problem index > midpoint, returns data before that
            elif problem_idx > midpoint:
                track = track.isel(time=slice(None, problem_idx))
                
        except IndexError:
            return track

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
        else:
            if 'coastline' in features.keys():
                ax.add_feature(cfeature.COASTLINE.with_scale(res), **coastline_kw)
            if 'states' in features.keys():
                ax.add_feature(cfeature.STATES.with_scale(res), **states_kw)
            if 'lakes' in features.keys():
                ax.add_feature(cfeature.LAKES.with_scale(res), **lakes_kw)
            if 'ocean' in features.keys():
                ax.add_feature(cfeature.OCEAN.with_scale(res), **ocean_kw)
                
    def plot_extras(self, fig, name, savefig=False, out_path=None):
        """
        Function to automatically plot title and legend based on 
        supplied name parameter.
        """
        # Checks to see if both savefig and out_path are adequately supplied
        if savefig == True and not out_path:
            raise ValueError('Must supply `out_path`.')
        elif savefig == True and not isinstance(out_path, str):
            raise TypeError('Must supply `out_path` as string.')
        elif not isinstance(savefig, bool):
            raise TypeError('Must supply `savefig` as bool.')
        elif savefig == False and out_path:
            savefig = True
        
        # Checks to see if supplied plot name is among available options
        names = ['spaghetti_all', 'miami_spaghetti', 'selected_tracks', 'overlaid_tracks']
        if name not in names:
            raise ValueError(f'Supplied `name` not in available options: {names}')
        elif name == 'spaghetti_all':
            suptitle = 'All Tracks Identified by TempestExtremes'
            out_file = 'te_spaghetti_all.png'
        elif name == 'miami_spaghetti':
            suptitle = 'Potential Tracks Within 500 km of Miami'
            out_file = 'te_spaghetti_miami.png'
        elif name == 'selected_tracks':
            suptitle = 'Selected Tracks for Dynamical Downscaling'
            out_file = 'selected_tracks.png'
        elif name == 'overlaid_tracks':
            suptitle = 'Simulated Storm Tracks'
            out_file = 'simulated_tracks.png'
        elif name == 'historical_tracks':
            suptitle = 'Historical Storm Tracks'
            out_file = 'historical_tracks.png'
            
        # Adds figure title
        if 'spaghetti' in name:
            fs = 16
        elif 'tracks' in name:
            fs = 14
        fig.suptitle(suptitle, fontweight='bold', fontsize=fs)
            
        # Retrieves SSHS labels and hex colors
        cat_mags = [160, 135, 115, 100, 75, 50, 35]
        cats = SaffirSimpson(units='mph', var=cat_mags)
        cat_labels = cats.category
        cat_colors = cats.color
            
        # Adds legend outside of plot
        if 'spaghetti' in name:
            lw = 2.0
            style_dict = {'xdata':[], 'ydata':[], 'ls':'-', 'lw':lw, 
                          'path_effects':[pe.Stroke(linewidth=lw*1.5, foreground='#818181'), pe.Normal()]}
            cat_handles = [mlines.Line2D(label=l, color=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]
        elif 'tracks' in name:
            mew = 0.75      # marker edge width
            mec = 'k'      # marker edge color
            ms = 6         # marker size
            lw = 1.0       # segment line weight
            style_dict = {'xdata':[], 'ydata':[], 'ls':'-', 'lw':lw, 'marker':'o', 'ms':ms, 'mew':mew, 'color':'k'}
            cat_handles = [mlines.Line2D(label=l, mfc=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]
        
        legend_kw = {'handles':cat_handles, 'loc':'outside lower center', 'fontsize':12, 'handlelength':2.0, 'ncols':3}
        
        if 'tracks' in name:
            legend_kw['fontsize'] = 11
        
        fig.legend(**legend_kw, title='Modified Saffir-Simpson Hurricane Scale', title_fontproperties={'weight':'bold', 'size':12})
        
        # Saves figure if desired
        if savefig:
            fig.savefig(os.path.join(out_path, out_file), dpi=300, bbox_inches='tight', transparent=False)
            plt.close()
        elif not savefig:
            plt.show()

class Plot():
    
    def __init__(storm_dict=None, savefig=False):
        
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
            lw = 2.0
            lw_e = 3.0
            style_dict = {'xdata':[], 'ydata':[], 'ls':'-', 'lw':lw, 
                          'path_effects':[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()]}
            cats_handles = [mlines.Line2D(label=l, color=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]

        elif feat_type == 'points':   
            # Marker properties
            mew = 0.5      # marker edge width
            mec = 'k'      # marker edge color
            ms = 8         # marker size
            #mfc           # marker face color

            style_dict = {'xdata':[], 'ydata':[], 'marker': 'o', 'ms':ms, 
                          'mew': mew, 'color':'k', 'ls':'-', 'lw': 0.5}
            cats_handles = [mlines.Line2D(label=l, mfc=c, **style_dict) for (l, c) in zip(cat_labels, cat_colors)]

        l = ax.legend(handles=cats_handles, loc='upper right', fontsize=14, shadow=True)
        l.set_zorder(1001)

    def plot_track(self, ax, data, feat_type='points', proj=ccrs.PlateCarree(), unit='pressure', legend=True, **kwargs):
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
        """

        # If kwargs aren't supplied, set defaults
        figtitle = kwargs.pop('figtitle', None)
        points_kw = kwargs.pop('point_kw', {})
        segments_kw = kwargs.pop('segments_kw', {})
        title_kw = kwargs.pop('title_kw', {})

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
                    points_style = {'edgecolors':'k', 'lw':0.5, 's':75, 'transform':proj, 'zorder':10}
                    points_kw = {**points_style, **points_kw}

                    # Plots track points
                    ax.scatter(lons, lats, c=sshws_cmap, **points_kw)

                    # Updates kwargs in place, prioritizing supplied kwargs over defaults
                    segments_style = {'zorder': 9, 'transform':proj, 'lw':0.5, 'ls':'-', 'colors':'k'}
                    segments_kw = {**segments_style, **segments_kw}

                elif feat_type == 'segments':
                    # Updates kwargs in place, prioritizing supplied kwargs over defaults
                    segments_style = {'zorder':10, 'transform':proj, 'lw': 1.25, 'colors': sshws_cmap, 
                                      'path_effects': [pe.Stroke(linewidth=2.0, foreground='#848484'), 
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
                segments_style = {'zorder':10, 'transform':proj, 'lw': 1.25, 'colors': sshws_cmap, 
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
            title_style = {'fontsize':20, 'fontstyle':'bold'}
            title_kw = {**title_style, **title_kw}
            plt.title(figtitle, **title_kw)

        if savefig == True:
            plt.show()
        else:
            plt.show()
    
    def plot_histograms(self, storm_name, bbox, variable, plot_regridded=False, **kwargs):
        """
        Histograms for precipitation and wind data. Actually faster 
        (1/2 the time) to calculate the Gaussian PDF and 
        plot in MPL than it is to use seaborn's histplot w/KDE.

        Parameters
        ----------------------

        variable  :: str, list, np.ndarray - target variable(s) for plotting
        """

        # Retrieves appropriate datasets from huge storm dict
        precip_vars = ['PRECT_TOT', 'PRECT_MAX_RATE']
        wind_vars = ['U10_MAX', 'WSP850_MAX']
        if variable in precip_vars or variable == precip_vars:
            if plot_regridded == True:
                ds_arr = ['h4pn_ds', 'h4pr_ds', 'h4cn_ds', 'h4cr_ds']
            else:
                ds_arr = ['h4pn_ds', 'h4cn_ds']
        elif variable in wind_vars or variable == wind_vars:
            if plot_regridded == True:
                ds_arr = ['h3pn_ds', 'h3pr_ds', 'h3cn_ds', 'h3cr_ds']
            else:
                ds_arr = ['h3pn_ds', 'h3cn_ds']
        else:
            msg = f"`variable` not in acceptable options: {[*precip_vars, *wind_vars]}."
            raise ValueError(msg)

        # Packs target datasets into list
        ds_list = list(map(lambda x: self.data[storm_name][x], ds_arr))

        # Subsets data spatially
        subsets = list(map(spatial_subset, ds_list, [bbox]*len(ds_arr)))

        # Retrieves metric variables, packs them into a list for easier iteration
        if isinstance(variable, (list, np.array)):
            var1_list = list(map(lambda x: x[variable[0]].values.flatten(), subsets))
            var2_list = list(map(lambda x: x[variable[1]].values.flatten(), subsets))
            var_list = [item for sublist in zip(var1_list, var2_list) for item in sublist]
        else:
            var_list = list(map(lambda x: x[variable].values.flatten(), subsets))

        # Manually calculates histogram bins and gaussian x values
        bins_list = list(map(lambda x: np.histogram_bin_edges(x, bins='auto'), var_list))
        eval_pts = list(map(lambda x: np.linspace(np.min(x), np.max(x)), var_list))

        # Calculates gaussian curves
        gauss = list(map(calc_gauss, var_list, eval_pts))

        # Sets up plotting defaults
        plot_kw = kwargs.get('plot_kw', {})
        font_kw = kwargs.get('plot_kw', {})

        plot_style = {'cam_colors':['#024588', '#7a8fc2'], 'mpas_colors':['#FA03B2', '#ff8bd1'], 
                      'labels':['CAM (Native)', 'CAM (Regridded)', 'MPAS (Native)', 'MPAS (Regridded)'], 
                      'ylabs':['Density', None, 'Density', None]}

        plot_style['colors'] = [*plot_style['cam_colors'], *plot_style['mpas_colors']]

        if plot_regridded == False:
            plot_style['labels'] = plot_style['labels'][::2]
            plot_style['colors'] = plot_style['colors'][::2]
            plot_style['ylabs'] = plot_style['ylabs'][:2]
            plot_kw = {'ncols':2, 'nrows':1, 'figsize':(6, 3), 'layout':'constrained'}
        else:
            plot_kw = {'ncols':2, 'nrows':2, 'figsize':(6, 6), 'layout':'constrained'}

        plot_style['colors'] = [x for pair in zip(plot_style['colors'], 
                                                  plot_style['colors']) for x in pair]

        if variable == precip_vars:
            plot_style['xlab'] = [None, None, 'Accumulated Precipitation [inches]', 
                                  'Precipitation Rate [inches/hour]']
            plot_style['titles'] = ['Total Precipitation', 'Maximum Hourly Precipitation', 
                                    None, None]
        elif variable == wind_vars:
            plot_style['xlab'] = [None, None, 'Wind Speed [mph]', 'Wind Speed [mph]']
            plot_style['titles'] = ['Maximum Surface Wind Speed', 'Maximum 850 mb Wind Speed', 
                                    None, None]

        font_kw = {'titles':9, 'labels':7, 'ticks':6, 'legend':7}

        fig, axs = plt.subplots(dpi=300, **plot_kw)

        hist_style = {'lw':0.3, 'histtype':'step', 'density':True}
        gauss_style = {'lw':0.4, 'ls':'--'}

        for i, ax in enumerate(axs.ravel()):
            # Plots CAM first
            ax.hist(var_list[i], color=colors[i], label=labels[i], bins=eval_pts[i], **hist_style)
            ax.plot(eval_pts[i], gauss[i], c=colors[i], **gauss_style)

            # Then plots MPAS
            midpt = int(len(var_list)/2)
            ax.hist(var_list[midpt:][i], color=colors[midpt:][i], 
                    label=labels[midpt:][i], bins=eval_pts[midpt:][i], **hist_style)
            ax.plot(eval_pts[midpt:][i], gauss[midpt:][i], c=colors[midpt:][i], **gauss_style)

            ax.set_title(titles[i], fontsize=font_kw['titles'], fontweight='bold')
            ax.set_xlabel(xlabs[i], fontsize=font_kw['labels'])
            ax.set_ylabel(ylabs[i], fontsize=font_kw['labels'])
            ax.tick_params(axis='both', which='major', labelsize=fonts['ticks'])
            ax.legend(fontsize=fonts['legend'])

        if savefig == True:
            print(f'Saving figure for {track_name}.')
            #fig.savefig(f"../figs/{variable}_histograms_{track_name}.png"), bbox_inches='tight', dpi=300, transparent=False)
            plt.close()
        else:
            plt.show()
            
    def plot_ddf(self, data, storm_name, variable, bbox, rolling_windows=None, percentiles=[5, 50, 95], plot_regridded=False, savefig=False, **kwargs):
        """
        Function to plot exposure curves for precipitation or wind.
        """
    
        fig_kw = kwargs.get('fig_kw', {})
        plot_kw = kwargs.get('plot_kw', {})
        font_kw = kwargs.get('font_kw', {})
        
        # Retrieves appropriate datasets from huge storm dict
        if plot_regridded == True:
            fig_kw['ncols'] = 2
            fig_kw['nrows'] = 2
            if variable == 'PRECT':
                ds_arr = ['h4pn_ds', 'h4pr_ds', 'h4cn_ds', 'h4cr_ds']
            elif variable == 'U10':
                ds_arr = ['h3pn_ds', 'h3pr_ds', 'h3cn_ds', 'h3cr_ds']
        if plot_regridded == False:
            fig_kw['ncols'] = 2
            fig_kw['nrows'] = 1
            if variable == 'PRECT':
                ds_arr = ['h4pn_ds', 'h4cn_ds']
            elif variable == 'U10':
                ds_arr = ['h3pn_ds', 'h3cn_ds']
                
        if variable not in ['PRECT', 'U10', 'WSP850']:
            raise ValueError("Must supply 'PRECT' or 'U10' to `variable`.")
            
        # Sets default values for rolling_windows
        if not rolling_windows:
            rolling_windows = [1, 3, 6, 12, 24, 36, 48, 72, 96, 120]
                
        # Packs target datasets into list
        ds_list = list(map(lambda x: data[storm_name][x], ds_arr))
        
        # Retrieves maximum and percentile arrays
        max_arrs, percentile_arrs = zip(*map(calc_ddf, ds_list, [variable]*len(ds_arr), 
                                             [bbox]*len(ds_arr), [rolling_windows]*len(ds_arr)))
    
        # Takes median, 5th-, and 95th- percentiles
        p5, medians, p95 = zip(*percentile_arrs)
        
        # Sets up plotting defaults
        fig_kw = {**{'figsize':(6, 3.5), 'dpi':300}, **fig_kw}
        font_kw = {**{'labels':7, 'titles':8, 'ticks':5.5}, **font_kw}
        
        if variable == 'U10':
            plot_kw['title'] = f'Wind Exposure ({storm_name})'
            plot_kw['ylabel'] = 'Maximum Wind Speed [mph]'
        elif variable =='PRECT':
            plot_kw['title'] = f'Accumulated Precipitation ({storm_name})'
            plot_kw['ylabel'] = 'Accumulated Precipitation [in]'
            
        plot_kw = {**{'xlabel':'Hours of Exposure'}, **plot_kw}
        cam_colors = ['#024588', '#6ca3c1', '#3374a7', '#6ca3c1']
        mpas_colors = ['#FA03B2', '#fb9ee4', '#fb6bce', '#fb9ee4']
        colors = [cam_colors, mpas_colors]
        
        # Plots output
        fig, axs = plt.subplots(**fig_kw, sharey=True, layout='constrained')
        for i, ax in enumerate(axs.ravel()):
            x_axis = rolling_windows
            ax.plot(x_axis, max_arrs[i], label='Maximum', c=colors[i][0], lw=1.0)
            ax.plot(x_axis, p5[i], label='5th Percentile', ls=':', c=colors[i][1], lw=1.0)
            ax.plot(x_axis, medians[i], label='Median', ls='--', c=colors[i][2], lw=1.0)
            ax.plot(x_axis, p95[i], label='95th Percentile', ls=':', c=colors[i][3], lw=1.0)
            
            ax.set_title(['CAM5-SE', 'CAM5-MPAS'][i], fontsize=font_kw['titles'])
            ax.set_xlabel(plot_kw['xlabel'], fontsize=font_kw['labels'])
            if i == 0:
                ax.set_ylabel(plot_kw['ylabel'], fontsize=font_kw['labels'])
            ax.set_xticks(x_axis, x_axis)
            ax.tick_params(axis='both', labelsize=font_kw['ticks'])

            ax.legend(fontsize=font_kw['labels'])
        
        fig.suptitle(plot_kw['title'], fontsize=font_kw['titles']+2, fontweight='bold')
        if savefig == True:
            plt.close()
        else:
            plt.show()
    
    def plot_panels(self, storm_name, bbox, ds_name=None, variable=None, num_panels=1, plot_type='track', **kwargs):
        """
        Parameters
        ---------------------
        plot_type  :: str - options are 'track', 'histogram', 'ddf curves'
        num_panels :: int (opt), default=1 - number of mpl subplots
        save_fig   :: bool (opt), default=False - whether or not to save figure
        
        Optional kwargs
        ---------------------
        proj      :: ccrs, default=ccrs.PlateCarree() - geographic projection
        hist_kw   :: args/kwargs to pass to plot_histogram()
        ddf_kw    :: args/kwargs to pass to plot_ddf()
        ts_kw     :: args/kwargs to pass to plot_timeseries()
        fig_kw    :: args to pass to plt.subplots()
        geog_kw   :: args/kwargs to pass to geog_features()
        track_kw  :: args/kwargs to pass to plot_track()
        
        
        output_kw :: kwargs to pass to different plots such as file name
        """
        
        # If kwargs aren't supplied, set defaults
        fig_kw = kwargs.get('fig_kw', {'dpi':300})
        proj = kwargs.get('proj', ccrs.PlateCarree())
        geog_kw = kwargs.get('geog_kw', {'proj':proj})
        
        track_kw = kwargs.get('track_kw', {})
        ddf_kw = kwargs.get('ddf_kw', {})
        ts_kw = kwargs.get('ts_kw', {})
        hist_kw = kwargs.get('hist_kw', {'var':None, 'regridded':False})
        
        acceptable_plots = ['track', 'histogram', 'ddf', 'timeseries']
        if plot_type not in acceptable_plots:
            msg = f"Supplied `plot_type` not in acceptable options: {acceptable_plots}."
            raise ValueError(msg)
            
        if num_panels == 1:
            if plot_type == 'track':
                fig_style = {'figsize':(12, 7), 
                             'subplot_kw':dict(projection=proj)}
                fig_kw = {**fig_style, **fig_kw}
                fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kw)
                self.geog_features(ax, **geog_kw)
                data = self.data[storm_name][ds_name]
                self.plot_track(ax, data)
                
            #elif plot_type == 'histogram':
                
                
        elif num_panels == 2:
            if plot_type == 'track':
                fig, axs = plt.subplots(nrows=1, ncols=2, **fig_kw)
            #elif plot_type == 'histogram':
            
            #elif plot_type = 'ddf':
                
            
            
        elif num_panels == 3:
            if plot_type == 'track':
                fig_style = {'figsize':(17, 10), 
                             'subplot_kw':dict(projection=proj), 
                             'layout':'constrained'}
                fig_kw = {**fig_style, **fig_kw}
                
                fig, axs = plt.subplots(nrows=1, ncols=3, **fig_style)
                for ax in axs.ravel():
                    self.geog_features(ax, **geog_kw)
                    
        elif num_panels == 9:
            if plot_type == 'track':
                fig, axs = plt.subplots(3, 3, figsize=(12, 12), subplot_kw=dict(projection=proj), **fig_kw)
                for i, ax in enumerate(axs.ravel()):
                    self.geog_features(ax, **geog_kw)
                    data = self.data[self.data.keys()[i]][ds_name]
                    self.plot_track(ax, data)
            #elif plot_type == 'timeseries':
                


