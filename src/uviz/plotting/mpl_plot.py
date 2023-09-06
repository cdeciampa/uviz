import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from uviz.plotting.utils import SaffirSimpson, basin_bboxes


class Plot():
    def __init__(self, proj=ccrs.PlateCarree(), bbox=None):
        
        self.basemap = self.geog_features(proj)
    

def geog_features(ax, basin='north atlantic zoomed', resolution='10m'):
    lons, lats = basin_bboxes(basin)
    ax.set_extent([lons[0], lons[1], lats[0], lats[1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=1.3)
    #ax.add_feature(cfeature.BORDERS.with_scale(resolution), linewidth=0.5, edgecolor='#323232', zorder=3)
    ax.add_feature(cfeature.STATES.with_scale(resolution), linewidth=0.5, facecolor='#EBEBEB', edgecolor='#616161', zorder=1.2)
    ax.add_feature(cfeature.LAKES.with_scale(resolution), linewidth=0.5, facecolor='#e4f1fa', edgecolor='#616161', zorder=1.2)
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor='#e4f1fa', edgecolor='face', zorder=1.1)

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

    cats = list(map(mlines.Line2D, list()*6, list()*6, 
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