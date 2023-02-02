from haversine import inverse_haversine, Direction, Unit

from matplotlib.collections import LineCollection
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class CustomColorbars():
    """
    Class to store my custom outgoing longwave radiation and recipitation colormaps.
    """
    def __init__(self, target_cmap):
        
        self.out_cmap, self.out_levels, self.norm = self.get_cmap(target_cmap)
        
    def get_cmap(self, target_cmap):
        
        if target_cmap == 'FLUT':
            brightness_temps = np.array([-110, -92.1, -92, -80, -70, -60, -50, -42, -30, -29.9, -20, -10, 0, 10, 20, 30, 40, 57])
            levels = np.array(list(map(self.T_to_FLUT, brightness_temps)))
            fracs = levels-self.T_to_FLUT(-110, 'C')
            fracs = fracs/fracs[-1]

            flut_colors = ['#ffffff', '#ffffff', '#e6e6e6', '#000000', '#ff1a00', '#e6ff01', '#00e30e', '#010073', '#00ffff', 
                           '#bebebe', '#acacac', '#999999', '#7e7e7e', '#6c6c6c', '#525252', '#404040', '#262626', '#070707']
            colormap = mcolors.LinearSegmentedColormap.from_list('FLUT CIMSS', list(zip(fracs, flut_colors)), N=1200)
            norm = mcolors.Normalize(vmin=levels[0], vmax=levels[-1])
        
        elif target_cmap == 'nws_precip':
            nws_precip_colors = [
                "#ffffff",  # 0.00 - 0.01 inches  white
                "#4bd2f7",  # 0.01 - 0.10 inches  light blue
                "#699fd0",  # 0.10 - 0.25 inches  mid blue
                "#3c4bac",  # 0.25 - 0.50 inches  dark blue
                "#3cf74b",  # 0.50 - 1.00 inches  light green
                "#3cb447",  # 1.00 - 1.50 inches  mid green
                "#3c8743",  # 1.50 - 2.00 inches  dark green
                "#1f4723",  # 2.00 - 3.00 inches  darkest green
                "#f7f73c",  # 3.00 - 4.00 inches  yellow
                "#fbde88",  # 4.00 - 5.00 inches  weird tan
                "#f7ac3c",  # 5.00 - 6.00 inches  orange
                "#c47908",  # 6.00 - 8.00 inches  dark orange
                "#f73c3c",  # 8.00 - 10.00 inch  bright red
                "#bf3c3c",  # 10.00 - 15.00 inch  mid red
                "#6e2b2b",  # 15.00 - 20.00 inch  dark red
                "#f73cf7",  # 20.00 - 25.00 inch  bright pink
                "#9974e4",  # 25.00 - 30.00 inch  purple
                #"#404040",  # 30.00 - 40.00 inch  dark gray because of mpl
                "#c2c2c2"  # 30.00 - 40.00 inch  gray
                ]
            colormap = mcolors.ListedColormap(nws_precip_colors, 'nws_precip')
            levels = [0.0, 0.01, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
            #levels = [0.01, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(levels))
        else:
            raise ValueError("Must choose either 'FLUT' or 'nws_precip' colormap.")
        
        return colormap, levels, norm

    def T_to_FLUT(self, T, unit='C'):
        if unit == 'C':
            T += 273.15
        sigma = 5.6693E-8
        olr = sigma*(T**4)

        return olr

class SaffirSimpson():
    def __init__(self, var, units):
        self.units = units
        wsp_units = ['m/s', 'knots', 'kts', 'mph']
        mslp_units = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals', 'mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']
        
        if self.units in wsp_units:
            self.category = self.sshs_wsp(self.var, self.units)
        elif self.units in mslp_units:
            self.category = self.sshs_mslp(self.var, self.units)
            
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
        elif units == 'mph':
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
                                                                  else '#C464D9' if x == 'Category 5' else ''))))))(self.category)
        return color
    
def geog_features(ax, basin='north atlantic zoomed', resolution='10m'):
    lons, lats = basin_bboxes(basin)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=3)
    #ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=3)
    ax.add_feature(cfeature.STATES.with_scale(resolution), linewidth=0.5, facecolor='#EBEBEB', edgecolor='#616161', zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale(resolution), linewidth=0.5, facecolor='#e4f1fa', edgecolor='#616161', zorder=2)
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor='#e4f1fa', edgecolor='face', zorder=1)

def plot_sshws_segments(ax, df, figtitle=None):
    
    proj = ccrs.PlateCarree()
    for track, track_df in df.groupby('tempest_ID'):
    
        lons = track_df['lon'].values
        lats = track_df['lat'].values
        wsps = track_df['wsp'].values
        sshws_cmap = [sshws_color(x, units='m/s') for x in wsps]

        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, colors=sshws_cmap, zorder=10, transform=proj, 
                            lw=1.25, path_effects=[pe.Stroke(linewidth=2.0, foreground='#848484'), pe.Normal()])
        ax.add_collection(lc)
    
    lw = 2.0
    lw_e = 3.0
    td = mlines.Line2D([], [], ls='-', lw=lw, label='Tropical Depression', color=sshws_color(35, 'mph'),
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    ts = mlines.Line2D([], [], ls='-', lw=lw, label='Tropical Storm', color=sshws_color(50, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    c1 = mlines.Line2D([], [], ls='-', lw=lw, label='Category 1', color=sshws_color(75, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    c2 = mlines.Line2D([], [], ls='-', lw=lw, label='Category 2', color=sshws_color(100, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    c3 = mlines.Line2D([], [], ls='-', lw=lw, label='Category 3', color=sshws_color(115, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    c4 = mlines.Line2D([], [], ls='-', lw=lw, label='Category 4', color=sshws_color(135, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])
    c5 = mlines.Line2D([], [], ls='-', lw=lw, label='Category 5', color=sshws_color(160, 'mph'), 
                       path_effects=[pe.Stroke(linewidth=lw_e, foreground='#848484'), pe.Normal()])

    l = ax.legend(handles = [c5, c4, c3, c2, c1, ts, td], loc='upper right', fontsize=14, shadow=True)
    l.set_zorder(1001)
    plt.title(figtitle, fontsize=20)
    plt.show()
    
def plot_sshws_points(ax, df, figtitle=None, j=None, label_tracks=False):
    
    proj = ccrs.PlateCarree()
    for i, (track_ID, track_df) in enumerate(df.groupby('tempest_ID')):
        track_df = track_df.sort_values(by=['tempest_ID', 'time']).reset_index(drop=True)
        lons = track_df['lon'].values
        lats = track_df['lat'].values
        wsps = track_df['wsp'].values
        sshws_cmap = [sshws_color(x, units='m/s') for x in wsps]
        
        if j == 0:
            label_pos = [[lons[8]+1.25, lats[8]+0.2], [lons[13]+1.25, lats[13]+0.2], [lons[10]-1.25, lats[10]-0.2]]
        elif j == 1:
            label_pos = [[lons[13]-1.25, lats[13]+0.2], [lons[6]-1.25, lats[6]+0.2], [lons[8]-1.25, lats[8]-0.2], [lons[3]+1.25, lats[3]+0.2]]
        elif j == 2:
            label_pos = [[lons[13]-1.25, lats[13]-0.2], [lons[11]+1.25, lats[11]+0.2], [lons[-9]-1.25, lats[-9]-0.2], [lons[8]+1.25, lats[8]+0.2]]
        elif j == 3:
            label_pos = []
        elif j == 4:
            label_pos = [[lons[14]-1.25, lats[14]+0.2], [lons[17]+1.25, lats[17]+0.35], [lons[-10]-1.1, lats[-10]-0.35]]
        elif j == 5:
            label_pos = [[lons[11]+1.25, lats[11]+0.2]]
        elif j == 6:
            label_pos = []
        elif j == 7:
            label_pos = [[lons[4]+1.25, lats[4]+0.2], [lons[0]-1.25, lats[0]-0.2]]
        elif j == 8:
            label_pos = [[lons[10]-1.25, lats[10]+0.2], [lons[14]-1.25, lats[14]-0.2]]

        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, colors='k', zorder=9, transform=proj, lw=0.5, ls='--')
        ax.add_collection(lc)
        ax.scatter(lons, lats, c=sshws_cmap, zorder=10, edgecolors='k', lw=0.35, s=30)
        # This labels individual tracks within model runs
        if label_tracks == True:
            ax.text(label_pos[i][0], label_pos[i][1], track_ID, transform=proj, fontsize=7.5, 
                    path_effects=[pe.Stroke(linewidth=1.75, foreground='w'), pe.Normal()], clip_on=True, ha='center', va='center')
    
    # Marker properties
    mew = 0.25     # marker edge width
    mec = 'k'      # marker edge color
    ms = 6         # marker size
    #mfc           # marker face color
    
    td = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Tropical Depression', mfc=sshws_color(35, 'mph'), color='k', lw=0.5, ls='--')
    ts = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Tropical Storm', mfc=sshws_color(50, 'mph'), color='k', lw=0.5, ls='--')
    c1 = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Category 1', mfc=sshws_color(75, 'mph'), color='k', lw=0.5, ls='--')
    c2 = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Category 2', mfc=sshws_color(100, 'mph'), color='k', lw=0.5, ls='--')
    c3 = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Category 3', mfc=sshws_color(115, 'mph'), color='k', lw=0.5, ls='--')
    c4 = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Category 4', mfc=sshws_color(135, 'mph'), color='k', lw=0.5, ls='--')
    c5 = mlines.Line2D([], [], marker='o', ms=ms, mew=mew, mec=mec, label='Category 5', mfc=sshws_color(160, 'mph'), color='k', lw=0.5, ls='--')

    l = ax.legend(handles = [c5, c4, c3, c2, c1, ts, td], loc='upper right', 
                  fontsize=10, shadow=False)
    l.set_zorder(1001)
    plt.title(figtitle)
    #plt.show()

def basin_bboxes(basin_name):
    """
    Creates predefined bounding boxes if supplied correct name.
    """
    basin_name = basin_name.lower()
    basins = ['north atlantic', 'north atlantic zoomed', 'florida', 'south florida', 'miami', 
              'south atlantic', 'east pacific', 'west pacific', 'north indian', 'south indian', 
              'australia', 'south pacific', 'conus', 'east conus']
    
    if basin_name not in basins:
        raise ValueError(f'{basin_name} not in list of basins. Choose from {basins}.')
        
    if basin_name == 'north atlantic':
        west_coord = -105.0+360
        east_coord = -5.0+360
        north_coord = 70.0
        south_coord = 0.0
    elif basin_name == 'north atlantic zoomed':
        west_coord = -100.0+360
        east_coord = -15.0+360
        north_coord = 50.0
        south_coord = 7.5
    elif basin_name == 'florida':
        west_coord = -90.0+360
        east_coord = -72.5+360
        north_coord = 32.5
        south_coord = 20.0
    elif basin_name == 'south florida':
        west_coord = -84.0+360
        east_coord = -79.0+360
        north_coord = 30.0
        south_coord = 24.0
    elif basin_name == 'miami':
        miami_coords = (25.775163, -80.208615)
        west_coord = inverse_haversine(miami_coords, 100, Direction.WEST, unit=Unit.KILOMETERS)[1]+360
        east_coord = inverse_haversine(miami_coords, 100, Direction.EAST, unit=Unit.KILOMETERS)[1]+360
        north_coord = inverse_haversine(miami_coords, 100, Direction.NORTH, unit=Unit.KILOMETERS)[0]
        south_coord = inverse_haversine(miami_coords, 100, Direction.SOUTH, unit=Unit.KILOMETERS)[0]
    elif basin_name == 'south atlantic':
        west_coord = -105.0+360
        east_coord = -5.0+360
        north_coord = 65.0
        south_coord = 0.0
    elif basin_name == 'east pacific':
        west_coord = -180.0+360
        east_coord = -80.0+360
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
        east_coord = -120.0+360
        north_coord = 0.0
        south_coord = -65.0        
    elif basin_name == 'conus':
        west_coord = -130.0+360
        east_coord = -65.0+360
        north_coord = 50.0
        south_coord = 20.0        
    elif basin_name == 'east conus':
        west_coord = -105.0+360
        east_coord = -60.0+360
        north_coord = 48.0
        south_coord = 20.0
    else:
        raise ValueError("Supplied name not in given list.")
    
    lons = (west_coord, east_coord)
    lats = (south_coord, north_coord)
    
    return lons, lats
