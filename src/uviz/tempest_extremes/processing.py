import numpy as np
import pandas as pd

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
                        Note: currently only works if last 4 column names
                        denote the year, month, day, and hour. Also the
                        first two column names must be supplied and are
                        assumed to be inconsequential.

        Returns
        --------------
        pandas DataFrame
        """

        # Declares column names (defaults of TE) if none supplied
        if self.colnames == None:
            colnames = ['i', 'j', 'lon', 'lat', 'slp', 'wind', 'phi', 
                        'year', 'month', 'day', 'hour']
        elif not isinstance(colnames, (list, np.ndarray)):
            raise ValueError('Must supply a list of column names.')

        # Reads in file
        try:
            df = pd.read_csv(file, sep='\s+', names=colnames)
        except pd.errors.ParserError as err:
            msg = f'Number of supplied column names != number of columns in file{str(err)[30:]}'
            raise ValueError(msg)

        # Identifies individual year, month, day, hour columns
        dt_cols = df.columns[[-4, -3, -2, -1]]

        # Takes individual year, month, day, hour columns and transforms to pd.date_time column
        df['time'] = pd.to_datetime(df[dt_cols], errors='coerce')

        # Drops year, month, day, hour columns
        df = df.drop(dt_cols, axis=1)

        # Selects indices where a new track starts, assigns to array
        run_idx = df[df[df.columns[0]]=='start'].index.tolist()

        # Separates dataframe into individual dataframes, split on new tracks
        dfs = [df.iloc[run_idx[n]+1:run_idx[n+1]] for n in range(len(run_idx)-1)]

        # Drops i and j columns
        dfs = [dfi.drop(dfi.columns[[0, 1]], axis=1) for dfi in dfs]

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
    

    