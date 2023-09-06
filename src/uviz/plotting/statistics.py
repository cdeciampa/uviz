import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import gaussian_kde, skew, ks_2samp
from sklearn.neighbors import KernelDensity

from uviz.plotting.utils import PlotProcessing, basin_bboxes

class Statistics():
    
    def __init__(self, model_data=None, nexrad_data=None, **kwargs):
        
        """
        Parameters
        ---------------
        model_data      :: dict
        nexrad_data     :: list of xr.Datasets
        
        Optional kwargs
        ----------------
        rolling_windows :: list, x-hourly rolling windows for DD curves
        percentiles     :: list, percentiles for
        overland        :: bool, whether or not to only consider grid cells
                            over land
        """
        
        # Only runs all of this if model_data and nexrad_data are both supplied
        if model_data and nexrad_data:
            
            # Defines default parameters
            rw = kwargs.get('rolling_windows', [1, 3, 6, 12, 24, 36, 48, 72, 96, 120])
            percs = kwargs.get('percentiles', [5, 50, 95])
            overland = kwargs.get('overland', True)
        
            # Declares/retrieves storm names
            storm_names = [x for x in model_data.keys() if x != 'Charley']
            
            # Retrieves precipitation datasets
            parent_ds = [model_data[x]['h4pn_ds'] for x in storm_names]
            child_ds = [model_data[x]['h4cn_ds'] for x in storm_names]
            
            # Ensures order of nexrad_data matches order of historical storms in model_data
            hist_names = ['Ian', 'Irma', 'Isaac', 'Fay']
            hist_order = [storm_names.index(hn) for hn in hist_names]
            hist_years = [2022, 2017, 2012, 2008]
            hist_years = [hist_years[i] for i in hist_order]
            
            nexrad_years = [ds.time[0].dt.year for ds in nexrad_data]
            nexrad_order = [nexrad_years.index(y) for y in hist_years]
            nexrad_data = [nexrad_data[i] for i in nexrad_order]
        
            # Declares empty dictionary (to be filled below)
            self.stats = {}
            
            # Loops through storm names to assign metrics to stats_dict
            for i, sn in enumerate(storm_names):
                
                # Retrieves bounding box (per storm)
                bbox = self.stats_bboxes(sn)
                
                print('Calculating DD curves...')
                p_dd_max, p_dd_percs = self.calc_dd(parent_ds[i], 'PRECT', bbox)
                c_dd_max, c_dd_percs = self.calc_dd(child_ds[i], 'PRECT', bbox)
                
                # For explicit indexing of nexrad_data
                if sn in hist_names:
                    j = 0
                    n_dd_max, n_dd_percs = self.calc_dd(nexrad_data[j])
                    
                    # Updates dictionary with NEXRAD calcs
                    self.stats[sn]['dd_metrics'].update{'n_ddf_max':n_dd_max}
                    self.stats[sn]['dd_metrics'].update{'n_ddf_percs':n_dd_percs}
                    j += 1
                
                # Updates dictionary
                self.stats[sn].update{'bbox':bbox}
                self.stats[sn]['dd_metrics'].update{'p_dd_max':p_dd_max}
                self.stats[sn]['dd_metrics'].update{'p_dd_percs':p_dd_percs}
                self.stats[sn]['dd_metrics'].update{'c_dd_max':c_dd_max}
                self.stats[sn]['dd_metrics'].update{'c_dd_percs':c_dd_percs}
                
            # Breaking loops up by metric to try to avoid running out of RAM
            for i, sn in enumerate(storm_names):
                
                # Retrieves bounding box (per storm)
                bbox = stats[sn]['bbox']
                
                print('Calculating histogram metrics...')
                # For total precipitation
                p_tot_arr, p_tot_bins, p_tot_gauss =\
                 self.get_hist_metrics(parent_ds[i], 'PRECT_TOT', bbox, rw, percs)
                c_tot_arr, c_tot_bins, c_tot_gauss =\
                 self.get_hist_metrics(child_ds[i], 'PRECT_TOT', bbox, rw, percs)
                
                # For maximum hourly rate of precipitation
                p_max_arr, p_max_bins, p_max_gauss =\
                 self.get_hist_metrics(parent_ds[i], 'PRECT_MAX', bbox, rw, percs)
                c_max_arr, c_max_bins, c_max_gauss =\
                 self.get_hist_metrics(child_ds[i], 'PRECT_MAX', bbox, rw, percs)
                
                # For explicit indexing of nexrad_data
                if sn in hist_names:
                    j = 0
                    # For total precipitation
                    n_tot_arr, n_tot_bins, n_tot_gauss =\
                     self.get_hist_metrics(nexrad_data[j], 'PRECT_TOT', bbox, rw, percs)
                    
                    # For maximum hourly rate of precipitation
                    n_max_arr, n_max_bins, n_max_gauss =\
                     self.get_hist_metrics(nexrad_data[j], 'PRECT_MAX', bbox, rw, percs)
                    
                    # Updates dictionary with NEXRAD calcs
                    self.stats[sn]['hist_metrics'].update{'n_tot_arr':n_tot_arr}
                    self.stats[sn]['hist_metrics'].update{'n_tot_bins':n_tot_bins}
                    self.stats[sn]['hist_metrics'].update{'n_tot_gauss':n_tot_gauss}
                    
                    self.stats[sn]['hist_metrics'].update{'n_max_arr':n_max_arr}
                    self.stats[sn]['hist_metrics'].update{'n_max_bins':n_max_bins}
                    self.stats[sn]['hist_metrics'].update{'n_max_gauss':n_max_gauss}
                    j += 1
                    
                # Updates dictionary with parent calcs
                self.stats[sn]['hist_metrics'].update{'p_tot_arr':p_tot_arr}
                self.stats[sn]['hist_metrics'].update{'p_tot_bins':p_tot_bins}
                self.stats[sn]['hist_metrics'].update{'p_tot_gauss':p_tot_gauss}
                self.stats[sn]['hist_metrics'].update{'p_max_arr':p_max_arr}
                self.stats[sn]['hist_metrics'].update{'p_max_bins':p_max_bins}
                self.stats[sn]['hist_metrics'].update{'p_max_gauss':p_max_gauss}
                
                # Updates dictionary with child calcs
                self.stats[sn]['hist_metrics'].update{'c_tot_arr':c_tot_arr}
                self.stats[sn]['hist_metrics'].update{'c_tot_bins':c_tot_bins}
                self.stats[sn]['hist_metrics'].update{'c_tot_gauss':c_tot_gauss}
                self.stats[sn]['hist_metrics'].update{'c_max_arr':c_max_arr}
                self.stats[sn]['hist_metrics'].update{'c_max_bins':c_max_bins}
                self.stats[sn]['hist_metrics'].update{'c_max_gauss':c_max_gauss}
                
            for i, sn in enumerate(storm_names):
                 # Retrieves bounding box (per storm)
                 bbox = stats[sn]['bbox']
             
                 print('Calculating cumulative density functions...')
                 # For total precipitation
                 p_tot_cdf = self.calc_cdf(stats[sn]['p_tot_arr'])
                 c_tot_cdf = self.calc_cdf(stats[sn]['c_tot_arr'])
                 
                 # For maximum hourly rate of precipitation
                 p_max_cdf = self.calc_cdf(stats[sn]['p_max_arr'])
                 c_max_cdf = self.calc_cdf(stats[sn]['c_max_arr'])
                 
                 if sn in hist_names:
                     n_tot_cdf = self.calc_cdf(stats[sn]['n_tot_arr'])
                     n_max_cdf = self.calc_cdf(stats[sn]['n_max_arr'])
                     
                     # Updates dictionary with NEXRAD calcs
                     self.stats[sn]['cdf_metrics'].update{'n_tot_cdf':n_tot_cdf}
                     self.stats[sn]['cdf_metrics'].update{'n_max_cdf':n_max_cdf}
                     
                # Updates dictionary with parent calcs
                self.stats[sn]['cdf_metrics'].update{'p_tot_cdf':p_tot_cdf}
                self.stats[sn]['cdf_metrics'].update{'p_max_cdf':p_max_cdf}
                     
                # Updates dictionary with child calcs
                self.stats[sn]['cdf_metrics'].update{'c_tot_cdf':c_tot_cdf}
                self.stats[sn]['cdf_metrics'].update{'c_max_cdf':c_max_cdf}
                
            # Calculates Kolmogorov-Smirnov metric for historical storms
            p_ks_tot = list(map(self.calc_ks, self.stats[hist_name]['n_tot_arr'], 
            self.stats[hist_name]['p_tot_arr']))
            p_ks_max = list(map(self.calc_ks, self.stats[hist_name]['n_max_arr'], 
            self.stats[hist_name]['p_max_arr']))
            
            c_ks_tot = list(map(self.calc_ks, self.stats[hist_name]['n_tot_arr'], 
            self.stats[hist_name]['c_tot_arr']))
            c_ks_max = list(map(self.calc_ks, self.stats[hist_name]['n_max_arr'], 
            self.stats[hist_name]['c_max_arr']))
            
            # Packs KS results into pandas DataFrame
            df_idx = [hist_names[i] for i in hist_order]
            ks_df = pd.DataFrame({'Parent_Tot':p_ks_tot, 'Parent_Max':p_ks_max, 
            'Child_Tot':c_ks_tot, 'Child_Max':c_ks_max}, index=df_idx)
            
            # Updates dictionary with KS results
            self.stats.update{'ks_df':ks_df}
        
    def calc_dd(self, ds, variable, bbox, rolling_windows=None, percentiles=[5, 50, 95], overland=True):
    
        """
        Parameters
        ----------------
        ds              :: xr.Dataset
        variable        :: str, target variable from data. 
                            Only 'PRECT' or 'U10' allowed.
        bbox            :: str or dict, feed to basin_bboxes
        rolling_windows :: list of rolling windows
        percentiles     :: list (int), list of percentiles to calculate.
        
        Returns
        ---------------
        max_arr         :: np.ndarray, array of maximum values
        percentile_arr  :: list of np.ndarrays for each percentile
        """
        
        if not rolling_windows:
            rolling_windows = [1, 3, 6, 12, 24, 36, 48, 72, 96, 120]
    
        # For model output dataset
        if 'ncol' in ds.dims:
            # Gets indices to subset data by based on bbox and overland params
            sub_idx = PlotProcessing.get_subset_idx(ds, bbox, overland)
    
            # Subsets data by above indices
            subset = ds.isel(ncol=sub_idx)
        
            # Converts units
            if variable == 'PRECT':
                units = 'in/hour'
            elif variable == 'U10':
                units = 'mph'
                rolling_windows = [int(round(x/3, 0)) for x in rolling_windows] 
            else:
                raise ValueError("Only 'PRECT' or 'U10' allowed as options.")
        
            # Creates DataFrame from array of values
            da = subset[variable].metpy.convert_units(units)
            arr = da.values
            df = pd.DataFrame(arr)

            # Creates new DataFrame for maximum sums (precip) or minimum maxima (wsp)
            if variable == 'PRECT':
                metric_df = pd.DataFrame({f'sum_{i}': df.rolling(i).sum().max() for i in rolling_windows}).T
            elif variable == 'U10':
                metric_df = pd.DataFrame({f'max_{i*3}': df.rolling(i).min().max() for i in rolling_windows}).T
            
        # For NEXRAD dataset
        elif 'lon' in ds.dims:
            # Converts longitude to [0, 360]
            ds['lon'] = np.mod(ds['lon'] + 180, 360) + 180

            # Subsets data by bbox
            subset = PlotProcessing.spatial_subset(ds, bbox)

            # Converts to numpy array for easier calculations
            arr = subset['rain_hourly'].values

            # Reshapes into (time, ncol)
            ncols = len(subset['lon'])*len(subset['lat'])
            arr = arr.T.reshape(ncols, len(subset['time']))

            # Creates DataFrame from array of values
            df = pd.DataFrame(arr)

            # Transposes DataFrame (for... some reason...)
            df = df.T

            # Creates new DataFrame for maximum sums
            metric_df = pd.DataFrame({f'sum_{i}': df.rolling(i, min_periods=1).sum().max() for i in rolling_windows}).T
        
        # Extracts into numpy array for faster calculation
        metric_arr = metric_df.values

        # Takes percentiles, ignoring any nans
        percentile_arr = list(np.nanpercentile(metric_arr, percentiles, axis=1))

        # Takes maxima, ignoring any nans
        max_arr = np.nanmax(metric_arr, axis=1)

        return max_arr, percentile_arr

    def calc_gauss(self, arr, eval_pts, method='scipy', **kwargs):
        """
        Function to manually calculate Gaussian PDF
        for plotting histograms with MPL. Default 
        method is 'scipy', which is what seaborn uses.
        
        Parameters
        -----------------
        data     :: np.ndarray, 1-D array of values
        eval_pts :: np.ndarray, 1-D array of values 
        method   :: str (opt), default='scipy', method
                        of calculating gaussians.
                        Options are 'scipy' or 'sklearn'.
                        
        Optional kwargs
        ------------------
        kde_kw   :: kwargs to pass to scipy.gaussian_kde or
                        sklearn.neighbors.KernelDensity
        """
        
        # Converts to array
        if isinstance(arr, list):
            arr = np.array(arr)

        # Retrieves KDE calculation kwargs
        kde_kw = kwargs.get('kde_kw', {})

        if method == 'scipy':
            # Calculates KDE
            kde = gaussian_kde(arr, **kde_kw)
            
            # Fits KDE to Gaussian PDF
            gauss = kde.pdf(eval_pts)

        elif method == 'sklearn':
            # Unpacks kwargs
            kde_kw = {'bandwidth':1, 'kernel':'gaussian', **kde_kw}
            
            # Calculates KDE
            kde = KernelDensity(**kde_kw).fit(arr.reshape([-1, 1]))
            
            # Fits KDE to Gaussian PDF
            gauss = np.exp(kde.score_samples(eval_pts.reshape([-1, 1])))
            
        else:
            raise ValueError("`method` not in acceptable values: 'scipy', 'sklearn'.")

        return gauss
    
    def fit_gauss(self, arr):
        """
        Fits data to Gaussian PDF.
        
        Parameters
        ---------------
        arr :: np.ndarray, array of data points
        
        Returns
        ---------------
        gauss_pdf :: np.ndarray, Gaussian PDF
        """
        
        gauss_pdf = gaussian_kde(arr).pdf(np.sort(arr))

        return gauss_pdf
        
    def stats_bboxes(self, storm_name):
        """
        Defines bounding boxes used for statistical calculations.
        
        Parameters
        ---------------
        storm_name :: str, storm name taken from keys used in ModelData.storms
        
        Returns
        ---------------
        bbox :: list, list of coordinates in ((west, east), (south, north))
                format from [0, 360].
        """
        
        if not isinstance(storm_name, str):
            raise TypeError('`storm_name` must be supplied as string.')
        
        if storm_name == 'storm_0236':
            west = -82.733860
            east = -80.036716
            south = 25.121120
            north = 27.495256
        elif storm_name == 'storm_0755':
            west = -82.733860
            east = -80.036716
            south = 25.348799
            north = 27.777070
        elif storm_name == 'storm_0528':
            west = -82.733860
            east = -80.036716
            south = 25.121120
            north = 27.495256
        elif storm_name == 'storm_1048':
            west = -82.858682
            east = -80.036716
            south = 25.629735
            north = 28.508998
        elif storm_name == 'storm_1521':
            west = -82.733860
            east = -80.036716
            south = 25.121120
            north = 27.946685
        elif storm_name == 'storm_1354':
            west = -81.992647
            east = -80.036716
            south = 25.121120
            north = 26.700814
        elif storm_name == 'storm_1307':
            west = -82.515101
            east = -80.036716
            south = 25.121120
            north = 27.210623
        elif storm_name == 'storm_0310':
            west = -82.861571
            east = -80.036716
            south = 25.659481
            north = 28.412060
        elif storm_name == 'storm_1279':
            west = -82.733860
            east = -80.036716
            south = 25.121120
            north = 27.357253
        elif storm_name in ['Irma', 'Fay']:
            west = -82.335520
            east = -80.042506
            south = 25.137884
            north = 26.919435
        elif storm_name == 'Isaac':
            west = -82.335520
            east = -80.042506
            south = 25.137884
            north = 26.919435
        elif storm_name == 'Ian':
            west = -82.857622
            east = -81.090739
            south = 26.426490
            north = 28.165928
        else:
            storm_opts = ['storm_0236', 'storm_0755', 'storm_0528', 
            'storm_1048', 'storm_1521', 'storm_1354', 'storm_1307', 
            'storm_0310', 'storm_1279', 'Ian', 'Irma', 'Isaac', 'Fay']
            raise ValueError(f'Storm name not in list: {storm_opts}')
        
        bbox = basin_bboxes(dict(west_coord=west, east_coord=east, 
        south_coord=south, north_coord=north))
        
        return bbox

    def get_hist_metrics(self, ds, variable, bbox):
        """
        Function to gather histogram variable metrics:
        data array, bins array, and fitted Gaussian PDF.
        """
        
        if 'ncol' in ds.dims:
            # Subsets model data by overland cells within bounding box
            subset_idx = PlotProcessing.get_subset_idx(ds, bbox)
            subset_ds = ds.isel(ncol=subset_idx)
            
            # Converts values to np.ndarray
            arr = subset_ds[variable].values.flatten()

        elif 'lon' in ds.dims:
            # Converts longitude to [0, 360]
            ds['lon'] = np.mod(ds['lon'] + 180, 360) + 180
            
            # Subsets NEXRAD data by cells within bounding box
            subset_ds = PlotProcessing.spatial_subset(ds, bbox, False)
            
            # Converts values to np.ndarray
            arr = subset_ds['rain_hourly'].values
            
            # Removes any grid cells that are all nans across time
            arr = arr[:, ~np.isnan(arr).all(axis=0)]

            # Navigates around other nans
            if variable == 'PRECT_TOT':
                arr = np.nansum(arr, axis=0)
            elif variable == 'PRECT_MAX':
                arr = np.nanmax(arr, axis=0)
                
        # Defines histogram bin intervals
        bins = np.histogram_bin_edges(arr, bins='auto')
        
        # Fits Gaussian PDF
        gauss = self.fit_gauss(np.sort(arr))

        return arr, bins, gauss
    
    def calc_cdf(self, arr):
        """
        Function to calculate Cumulative Density Function.
        """
        
        x = np.sort(arr)
        cdf = 1*(np.arange(arr.shape[0])/(arr.shape[0]-1))
    
        return cdf

    def lognorm_cdf(self, arr):
        """
        Alternative function to calculate CDF using scipy.
        """
        
        x = np.sort(x)
        shape, loc, scale = scipy.stats.lognorm.fit(x)
        cdf = scipy.stats.lognorm.cdf(x, shape, loc, scale)
    
        return cdf
        
    def calc_ks(self, model_arr, nexrad_arr):
        """
        Function to calculate Kolmogorov-Smirnoff test
        for historical storms.
        """
        
        ks_result = ks_2samp(nexrad_arr, model_arr)
        
        return ks_result
        