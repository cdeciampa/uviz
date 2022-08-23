import numpy as np
import pandas as pd
from haversine import haversine_vector
#from sklearn.metrics.pairwise import haversine_distances

def read_tempest_ASCII(file, colnames):
    
    # Reads in file
    df = pd.read_csv(file, sep='\s+', names=colnames)
    
    # Takes individual year, month, day, hour columns and transforms to pd.date_time column
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')
    
    # Drops year, month, day, hour columns
    df = df[['time', 'lon_x', 'lat_y', 'lon', 'lat', 'slp', 'wsp', 'sfc_phi']]
    
    # Selects indices where a new track starts, assigns to array
    run_idx = df[df.lon_x=='start'].index.tolist()

    # Separates dataframe into individual dataframes, split on new tracks
    dfs = [df.iloc[run_idx[n]+1:run_idx[n+1]] for n in range(len(run_idx)-1)]
    
    # Drops lon_x and lat_y columns
    dfs = [dfi[['time', 'lon', 'lat', 'slp', 'wsp', 'sfc_phi']] for dfi in dfs]
    
    # Resets index from previous dataframe splits
    dfs = [dfi.reset_index(drop=True) for dfi in dfs]
    
    # Creates storm IDs
    for i, dfi in enumerate(dfs):
        dfi['tempest_ID'] = f'storm_{i}'
    
    # Merges dfs back together
    df_concat = pd.concat([dfi for dfi in dfs]).reset_index(drop=True)
    
    return df_concat

def read_tempest_csv(file):
    
    # Reads in file, swaps outrageous wind values with NaN
    df = pd.read_csv(file, na_values='14142.00000')
    
    # Strips leading space in most columns
    df = df.rename(columns=lambda x: x.strip())
    
    # Takes individual year, month, day, hour columns and transforms to pd.date_time column
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')
    
    # Drops year, month, day, hour, i, j, and phis columns
    df = df[['track_id', 'time', 'lon', 'lat', 'slp', 'wind']]
    
    # Adds 'storm' as prefix to track_id
    df['track_id'] = df['track_id'] = 'storm_' + df['track_id'].astype(str)
    
    # Renames track_id column for processing with ibtracs
    df = df.rename(columns={'track_id':'tempest_ID'})
    
    return df

def match_tracks(tempest_track, ibtracs_df):
    
    merged = pd.merge(tempest_track, ibtracs_df, on='time')   # This is an inner merge by default (only matched tracks are kept)
    
    # to avoid pandas freaking out when trying to calculate haversine distance metric
    if merged.empty:
        return merged
    else:
        merged['dist_km'] = merged.apply(lambda x: haversine_vector((x.lat, x.lon),
                                                                 (x.LAT, x.LON), 
                                                                 unit='km')[0], axis=1).round(4)
        merged = merged[merged['dist_km'].apply(lambda x: x <= 300)].reset_index(drop=True)
        return merged
    
# Calculate hits to misses ratio (aim for ~70% match rate) - Done, see below
    
def matched_tracks(matched_dfs):

    matched_df = pd.concat([df for df in matched_dfs]).reset_index(drop=True)

    # Fixes weird repeating decimal for pressure (consequence of tempest.wsp sig figs, probably)
    matched_df['PRES'] = matched_df['PRES'].round(2)

    # Moves tempest_ID column up to more easily differentiate between tempest (lowercase) and ibtracs (uppercase) cols
    matched_cols = matched_df.columns.tolist()
    matched_cols = [matched_cols[0]] + [matched_cols[6]] + matched_cols[1:6] + matched_cols[7:]
    matched_df = matched_df[matched_cols]

    return matched_df
    
def hit_to_miss(ibtracs_track, tempest_df):
    merged = pd.merge(ibtracs_track, tempest_df, on='time')
    if merged.empty:
        return 'miss'
    else:
        merged['dist_km'] = merged.apply(lambda x: haversine_vector((x.lat, x.lon),
                                                                 (x.LAT, x.LON), 
                                                                 unit='km')[0], axis=1).round(4)
        merged = merged[merged['dist_km'].apply(lambda x: x <= 300)].reset_index(drop=True)
        if merged.empty:
            return 'miss'
        else:
            return 'hit'
        
def pod(hit_miss_array):
    
    hits = np.char.count(hit_miss_array, 'hit').sum()
    misses = np.char.count(hit_miss_array, 'miss').sum()

    # Probability of detection (Stella)
    pod = hits/(hits+misses)
    
    return pod
    