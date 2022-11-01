import matplotlib.colors as mcolors
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def ssh_wsp(wsp, units):
    if units == 'm/s':
        if wsp <= 17.0:
            return 'Tropical Depression'
        elif 17.0 < wsp < 33:
            return 'Tropical Storm'
        elif 33.0 <= wsp <= 42.0:
            return 'Category 1'
        elif 42.0 < wsp <= 49.0:
            return 'Category 2'
        elif 49.0 < wsp <= 57.0:
            return 'Category 3'
        elif 57.0 < wsp <= 70.0:
            return 'Category 4'
        elif wsp > 70.0:
            return 'Category 5'
    elif units == 'knots' or units == 'kts':
        if wsp <= 33.0:
            return 'Tropical Depression'
        elif 33.0 < wsp < 64:
            return 'Tropical Storm'
        elif 64.0 <= wsp <= 82.0:
            return 'Category 1'
        elif 82.0 < wsp <= 95.0:
            return 'Category 2'
        elif 95.0 < wsp <= 112.0:
            return 'Category 3'
        elif 112.0 < wsp <= 136.0:
            return 'Category 4'
        elif wsp > 136.0:
            return 'Category 5'
    elif units == 'mph':
        if wsp <= 38.0:
            return 'Tropical Depression'
        elif 38.0 < wsp < 74.0:
            return 'Tropical Storm'
        elif 74.0 <= wsp <= 95.0:
            return 'Category 1'
        elif 95.0 < wsp <= 110.0:
            return 'Category 2'
        elif 110.0 < wsp <= 129.0:
            return 'Category 3'
        elif 129.0 < wsp < 157.0:
            return 'Category 4'
        elif wsp >= 157.0:
            return 'Category 5'
    else:
        raise ValueError("Wind speed units must be 'm/s', 'kts', or 'mph'.")
        
# According to Klotzbach et. al., 2020
# Modified for TS and TD designation using Kantha, 2006
def ssh_mslp(slp, unit='pascal'):
    pascals = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals']
    hPa = ['mb', 'milibars', 'hPa', 'hectopascals', 'Hectopascals']
    
    if unit in pascals:
        slp = slp/100
    
    if slp <= 925.0:
        return 'Category 5'
    elif 946.0 >= slp > 925.0:
        return 'Category 4'
    elif 960.0 >= slp > 946.0:
        return 'Category 3'
    elif 975.0 >= slp > 960.0:
        return 'Category 2'
    elif 990.0 >= slp > 975.0:
        return 'Category 1'
    elif 1000.0 > slp > 990.0:
        return 'Tropical Storm'
    elif slp >= 1000.0:
        return 'Tropical Depression'

def sshws_color(var, units):
    wsp_units = ['m/s', 'knots', 'kts', 'mph']
    pres_units = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals', 'mb', 'milibars', 'hPa', 'hectopascals', 'Hectopascals']
    if units in wsp_units:
        category = ssh_wsp(var, units)
    elif units in pres_units:
        category = ssh_mslp(var, units)
    
    if category == 'Tropical Depression':
        return '#5EBAFF'
    elif category == 'Tropical Storm':
        return '#00FAF4'
    elif category == 'Category 1':
        return '#FFF795'
    elif category == 'Category 2':
        return '#FFD821'
    elif category == 'Category 3':
        return '#FF8F20'
    elif category == 'Category 4':
        return '#FF6060'
    elif category == 'Category 5':
        return '#C464D9'
    
def geog_features(ax, basin='north atlantic zoomed', resolution='10m'):
    lons, lats = basin_bboxes(basin)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=3)
    #ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=3)
    ax.add_feature(cfeature.STATES.with_scale(resolution), linewidth=0.5, facecolor='#EBEBEB', edgecolor='#616161', zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale(resolution), linewidth=0.5, facecolor='#e4f1fa', edgecolor='#616161', zorder=2)
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor='#e4f1fa', edgecolor='face', zorder=1)

def nonlinear_colorbar(var_name=None):
    if var_name == 'PRECIP':
        colors = [
                '#ffffff',  # 0 inches
                "#04e9e7",  # 0.01 - 0.10 inches
                "#019ff4",  # 0.10 - 0.25 inches
                "#0300f4",  # 0.25 - 0.50 inches
                "#02fd02",  # 0.50 - 0.75 inches
                "#01c501",  # 0.75 - 1.00 inches
                "#008e00",  # 1.00 - 1.50 inches
                "#fdf802",  # 1.50 - 2.00 inches
                "#e5bc00",  # 2.00 - 2.50 inches
                "#fd9500",  # 2.50 - 3.00 inches
                "#fd0000",  # 3.00 - 4.00 inches
                "#d40000",  # 4.00 - 5.00 inches
                "#bc0000",  # 5.00 - 6.00 inches
                "#f800fd",  # 6.00 - 8.00 inches
                "#9854c6",  # 8.00 - 10.00 inches
                "#fdfdfd"   # 10.00+
            ]
        levels = [0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
              6.0, 8.0, 10.]
            
        return cmap
    else:
        raise ValueError('fix me.')
        
    cmap = mcolors.ListedColormap(colors, var_name)    
    norm = mcolors.BoundaryNorm(levels, len(levels))
    
    return cmap

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
        west_coord = -83.0+360
        east_coord = -80.0+360
        north_coord = 29.0
        south_coord = 24.5
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
    
    lons = (west_coord, east_coord)
    lats = (south_coord, north_coord)
    
    return lons, lats
