import numpy as np
import pandas as pd
from haversine import haversine

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from uviz.tempest_extremes.processing import TempestExtremes
from uviz.tempest_extremes.ibtracs import IBTrACS

class TrackMetrics(TempestExtremes, IBTrACS):
    """
    Class for comparing TempestExtremes and IBTrACS.
    
    Parameters
    ----------------
    te_file :: path to TempestExtremes file (ASCII or .csv)
    ib_file :: path to IBTrACS file (.csv only)
    """
    
    def __init__(self, te_file, ib_file):
        print('Reading in and cleaning up TE file.')
        self.te_df = TempestExtremes(te_file).df
              
        print('Reading in IBTrACS file.')
        ibtracs = IBTrACS(ib_file)
        self.unclean_ib = ibtracs.unclean
        
        print('Cleaning up IBTrACS file.')
        self.clean_ib = ibtracs.clean
        
        print('Matching TE tracks to IBTrACS.')
        self.matched_unclean = self.match_tracks(self.te_df, self.unclean_ib)
        self.matched_clean = self.match_tracks(self.te_df, self.clean_ib)
    
    def match_tracks(self, te_df, ib_df, max_dist=300, unit='km'):
        """
        Function to match TempestExtremes identified tracks from 
        reanalysis data (ERA5) to IBTrACS database (.csv file only).

        Parameters
        --------------
        tempest_df :: TempestExtremes track dataframe
        ib_df      :: IBTrACS dataframe from IBTrACS.unclean or IBTrACS.clean
        """
        
        # Merges TempestExtremes and IBTrACS dataframes on datetime columns
        mdf = pd.merge(te_df, ib_df, on='time')
        
        # Converts lons from [0 ,360] to [-180, 180] (required for haversine)
        mdf['lon180'] = np.mod(mdf['lon'] - 180, 360) - 180
        mdf['LON180'] = np.mod(mdf['LON'] - 180, 360) - 180
        
        # Calculates haversine distances
        dists = mdf.apply(lambda x: haversine([x.lat, x.lon180], 
                                              [x.LAT, x.LON180], unit=unit), axis=1).round(4)
        mdf['dist'] = dists
        
        # Groups merged dataframe by minimum distances between TE and IBTrACS
        mdf_grouped = mdf.groupby(['time', 'tempest_ID'])['dist'].min().reset_index()
        
        # Merges grouped dataframe with original merged dataframe
        closest_df = mdf_grouped.merge(mdf, on=['tempest_ID', 'time', 'dist'], how='left')
        
        # Limits resulting dataframe to tracks within declared `max_dist`
        final_df = closest_df[closest_df['dist'] <= max_dist]
        
        return final_df
    
    def calc_far(self):
        """
        Function to calculate False Alarm Rate for TempestExtremes.
        """
        
        # Matches tracks based on "unclean" IBTrACS data
        mdf = self.matched_unclean
        
        # Calculates hits (number of TE tracks matched to IBTrACS)
        hits = len(mdf['tempest_ID'].unique().tolist())
        
        # Calculates false alarms (number of TE tracks not present in IBTrACS)
        false_alarms = len(self.te_df['tempest_ID'].unique().tolist()) - hits
        
        # Calculates false alarm rate (rate of mis-identified tracks found in TE)
        far = false_alarms/(hits+false_alarms)
        
        return far

    def check_fa_bias(self):
        """
        Function to check if there is a spatial false alarm
        bias (in the Southern Hemisphere)
        """
        
        # Retrieves dataframe of matched trackes based on unclean_ib
        mdf = self.matched_unclean
        
        # Identifies "hit" track IDs from matched tracks
        hits = mdf['tempest_ID'].unique().tolist()
        
        # Isolates false alarm tracks from original TempestExtremes dataframe
        fa_df = self.te_df[~self.te_df['tempest_ID'].isin(hits)]
        
        # Creates new column based on which hemisphere each FA is located
        fa_df = fa_df.assign(hemisphere=np.where(fa_df['lat'] > 0.0, 'northern', 'southern'))

        # Groups by hemisphere, counts how many false alarm tracks are there
        fa_grouped = fa_df.groupby(['hemisphere'])['tempest_ID'].size().reset_index('tracks')
        
        return fa_grouped
    
    def calc_pod(self):
        """
        Function to calculate Probability of Detection Rate for TempestExtremes.
        """
        
        # Matches tracks based on "clean" IBTrACS data
        mdf = self.matched_clean
        
        # Calculates hits (number of TE tracks matched to IBTrACS)
        hits = len(mdf['tempest_ID'].unique().tolist())
        
        # Calculates misses (number of IBTrACS tracks not present in TE)
        ib_IDs = len(self.clean_ib['SID'].unique().tolist())
        misses = ib_IDs - hits
        
        # Calculates PoD rate (number of correctly identified tracks found by TE)
        pod = hits/(hits+misses)
        
        return pod
    
    def calc_r2_rmse(self, performance='overall', norm=False):
        """
        Function to calculate the R^2 and Root Mean Square Errors
        for minimum sea level pressure and maximum wind speed for
        each track.

        Parameters
        -------------------
        performance :: str (opt), options are 'overall' or 'per track'
                            Chooses whether metrics are performed on a 
                            per-track basis or overall.
        norm :: boolean (opt), default=False - chooses whether or not to
                            min/max normalize data before performing
                            metrics (not recommended).

        Returns :: pd.DataFrame of R^2 and RMSE
        """

        # Retrieves dataframe, using clean IBTrACS data since pressure and
        # wind columns have already been collapsed into one column each.
        df = self.matched_clean

        # Normalizes values (not recommended, does weird things to R2 calcs)
        if norm == True:
            target_vars = df[['PRES', 'slp', 'WSP', 'wind']].values
            tvars_norm = MinMaxScaler().fit_transform(target_vars)
            df[['PRES', 'slp', 'WSP', 'wind']] = tvars_norm

        # Drops NaN values (there's only one: genesis WSP for GYAN, 1982)
        df = df.dropna()

        # Metrics are performed on a per-track basis
        if performance == 'per track':

            # Catches storms with only one track point
            point_df = df.groupby(['tempest_ID'])['PRES'].count().reset_index()
            point_tracks = point_df[point_df['PRES'] < 2]['tempest_ID'].values

            # Removes identified "point" tracks (otherwise R2 throws error for
            # being statistically insignificant).
            df = df[~df['tempest_ID'].isin(point_tracks)]

            # Calculates R2 and RMSE metrics per track
            gdf = df.groupby('tempest_ID')
            r2_pres = gdf.apply(lambda x: r2_score(x['PRES'], x['slp'])).values
            r2_wind = gdf.apply(lambda x: r2_score(x['WSP'], x['wind'])).values
            rmse_pres =  np.sqrt(gdf.apply(lambda x: mean_squared_error(x['PRES'], x['slp'])).values)
            rmse_wind =  np.sqrt(gdf.apply(lambda x: mean_squared_error(x['WSP'], x['wind'])).values)

            # Averages metrics for overall scores
            r2_pres = r2_pres.mean()
            r2_wind = r2_wind.mean()
            rmse_pres = rmse_pres.mean()
            rmse_wind = rmse_wind.mean()

        # Metrics are performed overall
        elif performance == 'overall':
            r2_pres = r2_score(df['PRES'], df['slp'])
            r2_wind = r2_score(df['WSP'], df['wind'])
            rmse_pres = np.sqrt(mean_squared_error(df['PRES'], df['slp']))
            rmse_wind = np.sqrt(mean_squared_error(df['WSP'], df['wind']))

        else:
            raise ValueError("Performance must == 'overall' or 'per track'.")

        # Packs metrics into a dataframe
        metrics_series = dict(R2_pres=r2_pres, R2_wind=r2_wind, 
                              RMSE_pres=rmse_pres, RMSE_wind=rmse_wind)
        metrics_df = pd.DataFrame(metrics_series, index=[0])

        return metrics_df