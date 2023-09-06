import os
import time
import argparse

from uviz.plotting.processing import ModelData
from uviz.plotting.utils import PlotProcessing
from uviz.plotting.mpl_plot import TrackPlot, StatsPlot
from uviz.plotting.statistics import Statistics

from uviz.tempest_extremes.processing import TempestExtremes
from uviz.tempest_extremes.ibtracs import IBTrACS


# Starts timer
start_time = time.process_time()

# Creates parser
parser = argparse.ArgumentParser()

# Arguments - all are optional to allow for use of defaults
parser.add_argument('-if', '--input-folder', 'input-folder', type=str, help='Root path to input data.', 
                    default='/gpfs/group/cmz5202/default/cnd5285')
parser.add_argument('-of', '--output-folder', 'output-folder', type=str, default='../figs', 
                    help='Path to output data (where figs go).')


parser.add_argument('-pl', '--parallel', 'parallel', action='store_false', 
                    help='Toggle parallelization for reading in model data.')


# Parses arguments
args = parser.parse_args()

# Prepares output folder
if not args.output_folder:
    print(f'Output folder not specified, using default: {args.output_folder}.')            
        if not os.path.isdir(args.output_folder):
            print('Output folder not found, creating directory.')
            os.mkdir(args.output_folder)
else:
    if not os.path.isdir(args.output_folder):
        print('Output folder not found, creating directory.')
        os.mkdir(args.output_folder)
        
# Creates first-level subfolders (if they don't already exist)
subfolders = ['sim_plots', 'panel_plots']
for sf in subfolders:
    try:
        os.mkdir(os.path.join(args.output_folder, sf))
    except FileExistsError:
        pass
        
# Ensures input folder exists
if not args.input_folder:
    print(f'Input folder not specified, using default: {args.input_folder}')
    if not os.path.isdir(args.input_folder):
        raise OSError(f'Default input folder not found: {args.input_folder}')
else:
    if not os.path.isdir(args.input_folder):
        raise OSError(f'Supplied input folder not found: {args.input_folder}')

########## Section that reads in data ##########
# Unstructured data
all_storms = ModelData(args.input_folder, args.parallel).storms

# NEXRAD data
print('Reading in NEXRAD datasets.')
irma_ds = xr.open_dataset('../data/irma_nexrad.nc')
ian_ds = xr.open_dataset('../data/ian_nexrad.nc')
isaac_ds = xr.open_dataset('../data/isaac_nexrad.nc')
fay_ds = xr.open_dataset('../data/fay_nexrad.nc')

# Ensures nexrad data timesteps == model data timesteps
ian_ds = ian_ds.sel(time=all_storms['Ian']['h4cn_ds'].time.astype('datetime64[ns]'))
irma_ds = irma_ds.sel(time=all_storms['Irma']['h4cn_ds'].time.astype('datetime64[ns]'))
isaac_ds = isaac_ds.sel(time=all_storms['Isaac']['h4cn_ds'].time.astype('datetime64[ns]'))
fay_ds = fay_ds.sel(time=all_storms['Fay']['h4cn_ds'].time.astype('datetime64[ns]'))

# IBTrACS
print('Reading in IBTrACS file.')
ib_df = IBTrACS('../tempest_extremes/ibtracs.since1980.list.v04r00.csv').clean

# Parent files (for track plots)
print('Reading in TempestExtremes track files.')
te_files = glob.glob('../tempest_extremes/trajectories.txt*VR28*')
te_dfs = [TempestExtremes(f).df for f in te_files]

# Creates second-level subfolders within `sim_plots`
storm_names = [s for s in all_storms.keys() if s != 'Charley']
for storm in storm_names:
    try:
        os.mkdir(os.path.join(args.output_folder, f'sim_plots/{storm}'))
    except FileExistsError:
        pass

########## Section that makes mesh figs? ##########
print('For the CAM5-SE and CAM5-MPAS global grid figures, please run the included NCL files.')

########## Section that makes track plots (spaghetti + selected tracks) ##########
# Declares figure panel labels and style
labs = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)']
labs_box_dict = dict(facecolor='white', edgecolor='k')
labs_dict = dict(x=0.04, y=0.925, verticalalignment='top', fontsize=12, 
                 fontweight='bold', bbox=labs_dict, zorder=100)

# Declares universal subplots style
subplots_kw = dict(dpi=300, layout='constrained', 
                   subplot_kw=dict(projection=ccrs.PlateCarree()))


##### Page 9 - 9-panel spaghetti plot of tracks < 500 km of Miami #####
potential_tracks = [miami_storms(df) for df in te_dfs]
spaghetti_tracks = [df[df['tempest_ID'].isin(x)].reset_index(drop=True) for df, 
                    x in zip(te_dfs, potential_tracks)]

# Organizes spaghetti tracks for Fig. 2.3 in thesis
subtitles = ['EXT.001', 'REF.001', 'WAT.001', 
             'EXT.002', 'REF.002', 'WAT.002', 
             'EXT.003', 'REF.003', 'WAT.003']

# Rearranges dataframes to match titles
tracks_i = [0, 2, 4, 1, 3, 5, 6, 7, 8]
spaghetti_tracks = [spaghetti_tracks[i] for i in tracks_i]

# Plots spaghetti tracks
fig, axs = plt.subplots(3, 3, figsize=(8, 6.25), **subplots_kw)

for i, ax in enumerate(axs.ravel()):
    
    # Plots tracks per ensemble run
    TrackPlot().plot_track(ax, spaghetti_tracks[i], 'north atlantic zoomed', 'segments', unit='pressure')
    ax.set_title(f'{subtitles[i]}', fontsize=14, fontweight='bold')
    
    # Adds panel labels
    ax.text(s=labs[i], transform=ax.transAxes, **labs_dict)

# Adds title and legend and also exports figure
print('Exporting spaghetti tracks within 500 km of Miami figure.')
TrackPlot().plot_extras(fig, 'spaghetti_miami', True, os.path.join(args.output_folder, 'panel_plots'))

##### Page 12 - 9-panel selected tracks plot #####
# Retrieves original tracks
selected_tracks = [['storm_1279'], [''], ['storm_1048'], 
                   ['storm_0236', 'storm_0528', 'storm_1521'], ['storm_0310'], ['storm_1354'], 
                   ['storm_1307'], ['storm_0755'], ['']]
selected_plots = [df[df['tempest_ID'].isin(x)].reset_index(drop=True) for df, x in zip(te_dfs, selected_tracks)]

# Orders them for consistency with later plots
ordered_dfs = [selected_plots[3][selected_plots[3]['tempest_ID'] == 'storm_0236'], 
               selected_plots[7], 
               selected_plots[3][selected_plots[3]['tempest_ID'] == 'storm_0528'], 
               selected_plots[2], 
               selected_plots[3][selected_plots[3]['tempest_ID'] == 'storm_1521'], 
               selected_plots[5], selected_plots[6], selected_plots[4], selected_plots[0]]

# Concatenates list of ordered dataframes into one dataframe
sel_df = pd.concat(ordered_dfs)

# Retrieves the subtitles in the same order
sim_names = [x for x in all_storms.keys() if x not in ['Ian', 'Irma', 'Isaac', 'Fay', 'Charley']]

# Sets up figure
fig, axs = plt.subplots(3, 3, figsize=(6.2, 6.5), **subplots_kw)

# Plots selected tracks
for i, ax in enumerate(axs.ravel()):
    
    # Plots track per storm
    TrackPlot().plot_track(ax, sel_df[sel_df['tempest_ID'] == sim_names[i]], 'florida', 'points', unit='pressure')
    ax.set_title(f'{sim_names[i]}', fontsize=12, fontweight='bold')
    
    # Adds labels
    ax.text(s=labs[i], transform=ax.transAxes, **labs_dict)

# Adds title and legend and also exports figure
print('Exporting tracks selected for dynamical downscaling figure.')
TrackPlot().plot_extras(fig, 'selected_tracks', True, os.path.join(args.output_folder, 'panel_plots'))
            
##### Page 17 - 12-panel historical tracks plot
# Declares historical storm names
hist_names = ['Ian', 'Irma', 'Isaac', 'Fay']
hist_years = [2022, 2017, 2012, 2008]
hist_subtitles = [f'{s} ({y})' for s, y in zip(hist_names, hist_years)]

# Retrieves historical track dataframes from IBTrACS
ian_df = ib_df[(ib_df['NAME'] == 'IAN') & (ib_df['SEASON'] == 2022)].reset_index(drop=True)
irma_df = ib_df[(ib_df['NAME'] == 'IRMA') & (ib_df['SEASON'] == 2017)].reset_index(drop=True)
isaac_df = ib_df[(ib_df['NAME'] == 'ISAAC') & (ib_df['SEASON'] == 2012)].reset_index(drop=True)
fay_df = ib_df[(ib_df['NAME'] == 'FAY') & (ib_df['SEASON'] == 2008)].reset_index(drop=True)

# Retrieves historical track points from parent simulations
hist_parent_tracks = [PlotProcessing().get_track_points(all_storms[x]['h3pn_ds'], 'PSL') for x in hist_names]

# Retrieves historical track points from child simulations
hist_child_tracks = [PlotProcessing().get_track_points(all_storms[x]['h3cn_ds'], 'PSL') for x in hist_names]

# Reorders tracks to be in order of Ian, Irma, Isaac, Fay
hist_parent_tracks = [parent_tracks[i] for i in [0, 1, 3, 2]]
hist_child_tracks = [child_tracks[i] for i in [0, 1, 3, 2]]

# Packs all historical tracks into list for plotting
hist_tracks = [*hist_parent_tracks, *hist_child_tracks, ian_df, irma_df, isaac_df, fay_df]

# Updates panel labels (for plotting)
labs_dict2 = dict(x=0.06, y=0.92, fontsize=9)
labs_dict = {**labs_dict, **labs_dict2}

# Declares historical storm plot subtitles
hist_subtitles = ['Ian (2022)', 'Irma (2017)', 'Isaac (2012)', 'Fay (2008)']

# Track plots (selected, modeled)
fig, axs = plt.subplots(3, 4, figsize=(6.2, 5), **subplots_kw)

# Adds labels
for i, ax in enumerate(axs.ravel()):
    TrackPlot().plot_track(ax, hist_tracks[i], 'florida', 'points', unit='pressure', points_kw=dict(s=16))
    
    # Plots subtitles
    if i in [0, 1, 2, 3]:
        ax.set_title(hist_subtitles[i], fontsize=12, fontweight='bold')
    
    # Plots panel labels
    ax.text(s=labs[i], transform=ax.transAxes, **labs_dict2)
    
# Adds title and legend and also exports figure
print('Exporting historical storm tracks.')
TrackPlot().plot_extras(fig, 'historical_tracks', True, os.path.join(args.output_folder, 'panel_plots'))

##### Page 31 - 9-panel overlaid child/parent tracks
# Retrieves simulated child tracks
sim_child_tracks = [PlotProcessing().get_track_points(all_storms[x]['h3cn_ds'], 'PSL') for x in sim_names]

# Updates panel label dictionary
labs_dict2 = dict(x=0.05, y=0.925, fontsize=10)
labs_dict = {**labs_dict, **labs_dict2}

# Sets up figure
fig, axs = plt.subplots(3, 3, figsize=(6.2, 6.5), **subplots_kw)

for i, ax in enumerate(axs.ravel()):
    
    # Plots parent simulation track points first
    TrackPlot().plot_track(ax, sel_df[sel_df['tempest_ID'] == sim_names[i]],
                           'florida', 'points', legend=False, unit='pressure', 
                           points_kw=dict(alpha=0.6, s=12), segments_kw=dict(alpha=0.5, lw=0.5))
    
    # Then plots child simulation track points on top
    TrackPlot().plot_track(ax, sim_child_tracks[i], 'florida', 'points', legend=False, 
                           unit='pressure', points_kw=dict(s=20, lw=0.6))
    
    # Plots subtitles
    ax.set_title(f'{sim_names[i]}', fontsize=12, fontweight='bold')
    
    # Adds panel labels
    ax.text(s=labs[i], transform=ax.transAxes, **labs_dict)

# Adds title and legend and also exports figure
print('Exporting overlaid child/parent simulated tracks figure.')
TrackPlot().plot_extras(fig, 'overlaid_tracks', True, os.path.join(args.output_folder, 'panel_plots'))

########## Section that makes total precip plots (with black bboxes) ##########
##### Page 18 - total precip for historical storms (8 individual holoviz plots + one 4-panel MPL plot)
    # MPL plot code is stored locally
##### Page 36 - 9 individual holoviz plots of parent output
##### Page 37 - 9 individual holoviz plots of child output

########## Section that makes maximum hourly precip plots (with black bboxes) ##########
##### Page 26 - max precip for historical storms (8 individual holoviz plots + one 4-panel MPL plot)
    # MPL plot code is stored locally
##### Page 42 - 9 individual holoviz plots of parent output
##### Page 43 - 9 individual holoviz plots of child output

########## Section that makes DD curves ##########
print('Beginning statistical calculations.')
nexrad_data = [ian_ds, irma_ds, isaac_ds, fay_ds]
stats_metrics = Statistics(all_storms, nexrad_data)

##### Page 30 - historical storms


##### Page 48 - simulated storms

########## Section that makes histograms ##########
##### Page 23 - historical total precip
##### Page 37 - historical maximum hourly precip
##### Page 40 - simulated total precip
##### Page 44 - simulated maximum hourly precip

########## Section that makes CDF curves ##########
##### Page 24 - historical total precip
##### Page 28 - historical max hourly precip
##### Page 29 - KS test results
print('Outputting Kolmogorov-Smirnoff test results table to .csv')
ks_df = stats_metrics['ks_df']


##### Page 41 - simulated total precip
##### Page 46 - simulated max hourly precip

########## Section that makes OLR figs ##########
##### Page 33 - simulated parent OLR
##### Page 34 - simulated child OLR

########## Section that makes OLR gif? ##########


end_time = time.process_time() - start_time
print(f'Figure output complete! Time elapsed: {end_time}')
