import pandas as pd
import numpy as np

class IBTrACS():
    def __init__(self, ibtracs_csv):
        
        self.file = ibtracs_csv
        self.unclean = self.prep_ibtracs()
        self.clean = self.clean_ibtracs()

    def prep_ibtracs(self):
        """
        Function that cleans the IBTrACS csv for processing. 
        Returns "unclean" dataframe for track comparison, 
        false alarm rate, and probability of detection (hits) metrics.
        
        Date last accessed: March 1, 2023

        Parameters
        -------------
        ibtracs_csv :: path to IBTrACS csv file

        Returns
        -------------
        pandas.DataFrame
        """

        # Non agency-specific ibtracs columns
        storm_cols = ['SID', 'SEASON', 'NAME', 'ISO_TIME', 'NATURE', 'LAT', 'LON', 'TRACK_TYPE']

        # Agency-specific pressure columns
        pres_cols = ['USA_PRES', 'WMO_PRES','TOKYO_PRES', 'CMA_PRES','HKO_PRES', 'NEWDELHI_PRES', 
                     'REUNION_PRES', 'BOM_PRES', 'NADI_PRES', 'WELLINGTON_PRES']
        # Declares as class variable for downstream processing
        self.pres_cols = pres_cols

        # Agency-specific max sustained wind speed columns
        wsp_cols = ['WMO_WIND', 'USA_WIND', 'TOKYO_WIND', 'CMA_WIND', 'HKO_WIND', 'NEWDELHI_WIND', 
                    'REUNION_WIND', 'BOM_WIND', 'NADI_WIND', 'WELLINGTON_WIND']

        # All selected ibtracs cols
        ibtracs_cols = storm_cols + pres_cols + wsp_cols
        nans = ['', ' ', -999, -99]
        
        # Declares file
        ibtracs_csv = self.file
        
        # Reads in file, assigns to dataframe
        unclean_df = pd.read_csv(ibtracs_csv, usecols=ibtracs_cols, skiprows=[1], 
                              dtype={'STORM_SPEED': np.float64, 'STORM_DIR': np.float64},
                              na_values=nans, parse_dates=['ISO_TIME'])

        # Renames time column, this is for simplified merging later on (could use right_on/left_on, but I didn't)
        unclean_df = unclean_df.rename(columns={'ISO_TIME':'time'})

        return unclean_df
    
    def clean_ibtracs(self):
        
        """
        Function that further cleans IBTrACS dataframe. Methods are
        defined in the comments. "Clean" dataframe is used for 
        probability of detection (misses) metric.
        
        Based off of Stella Bourdin's dynamicoPy code:
        https://github.com/stella-bourdin/dynamicoPy/blob/main/dynamicopy/tc/ibtracs.py
        """
        
        # Makes copy of unclean dataframe for processing
        clean_df = self.unclean.copy()

        # Drop anything that's not 0Z, 6Z, 12Z, or 18Z hour intervals
        clean_df = clean_df[clean_df['time'].apply(lambda x: x.hour in([0, 6, 12, 18]))].reset_index(drop=True)

        # Drop anything non-tropical
        # Possibly comment out this filter ?
        clean_df = clean_df[clean_df['NATURE']=='TS']

        # Drop 'spur' tracks (where the ibtracs algo confuses different tracks for the same storm)
        clean_df = clean_df[clean_df['TRACK_TYPE']=='main'].reset_index(drop=True)

        # Drop rows without WMO data
        clean_df = clean_df.dropna(subset=['WMO_PRES', 'WMO_WIND']).reset_index(drop=True)

        # Creates one pressure column based on Stella's code (prioritizes WMO values, 
        # otherwise averages over other agency pressure cols)
        clean_df['PRES'] = np.where(~clean_df.WMO_PRES.isna(), clean_df.WMO_PRES, 
                                   clean_df[self.pres_cols].mean(axis=1, skipna=True))

        # Converts pressure from mb/hPa to Pa
        clean_df['PRES'] = clean_df['PRES'].mul(100)

        # From Knapp and Kruk (2010) method for adjusting interagency wsp to 1-min sustained
        # https://doi.org/10.1175/2009MWR3123.1
        clean_df['TOKYO_WIND_1'] = clean_df.apply(lambda x: ((x['TOKYO_WIND'] - 23.3)/0.60), axis=1)
        clean_df['CMA_WIND_1'] = clean_df['CMA_WIND'].div(0.871)
        clean_df['HKO_WIND_1'] = clean_df['HKO_WIND'].div(0.9)
        clean_df['NEWDELHI_WIND_1'] = clean_df['NEWDELHI_WIND']
        clean_df['REUNION_WIND_1'] = clean_df['REUNION_WIND'].div(0.88)
        clean_df['BOM_WIND_1'] = clean_df['BOM_WIND'].div(0.88)
        clean_df['NADI_WIND_1'] = clean_df['NADI_WIND'].div(0.88)
        clean_df['WELLINGTON_WIND_1'] = clean_df['WELLINGTON_WIND'].div(0.88)
        clean_df['USA_WIND_1'] = clean_df['USA_WIND']

        # Creates one wsp column
        clean_df['wsp_kts'] = clean_df.filter(like='WIND_1').ffill(axis=1).iloc[:,-1]

        # Converts wsp column to m/s
        clean_df['WSP'] = clean_df['wsp_kts'].mul(0.514)

        # Drops all extra pressure columns
        clean_df = clean_df[['SID', 'SEASON', 'NAME', 'time', 'LAT', 'LON', 'PRES', 'WSP']]
        
        return clean_df