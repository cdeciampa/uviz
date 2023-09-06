import numpy as np
import pandas as pd
from haversine import haversine

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
        
        self.miami_df = self.miami_storms()
        
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

    def miami_storms(self, df, coords=(25.775163, -80.208615), (distance=500.0, units='km'):
        
        df['miami_dist'] = df.apply(lambda x: haversine((x.lat, x.lon), coords, 
                                                        unit=units, normalize=True), axis=1).round(2)
        df = df[df['miami_dist'].apply(lambda x: x <= distance)].reset_index(drop=True)
        storms = df['tempest_ID'].unique()

        return storms
    

    