import numpy as np
import xarray as xr

#ibtracs = "../tempest_extremes/IBTrACS.ALL.v04r00.nc"
#ibtracs_ds = xr.open_dataset(ibtracs, drop_variables = )

ibtracs_vars = [
    'tokyo_lat'          ,
    'tokyo_lon'          ,
    'tokyo_grade'        ,
    'tokyo_wind'         ,
    'tokyo_pres'         ,
    'tokyo_r50_dir'      ,
    'tokyo_r50_long'     ,
    'tokyo_r50_short'    ,
    'tokyo_r30_dir'      ,
    'tokyo_r30_long'     ,
    'tokyo_r30_short'    ,
    'tokyo_land'         ,
    'cma_lat'            ,
    'cma_lon'            ,
    'cma_cat'            ,
    'cma_wind'           ,
    'cma_pres'           ,
    'hko_lat'            ,
    'hko_lon'            ,
    'hko_cat'            ,
    'hko_wind'           ,
    'hko_pres'           ,
    'newdelhi_lat'       ,
    'newdelhi_lon'       ,
    'newdelhi_grade'     ,
    'newdelhi_wind'      ,
    'newdelhi_pres'      ,
    'newdelhi_ci'        ,
    'newdelhi_dp'        ,
    'newdelhi_poci'      ,
    'reunion_lat'        ,
    'reunion_lon'        ,
    'reunion_type'       ,
    'reunion_wind'       ,
    'reunion_pres'       ,
    'reunion_tnum'       ,
    'reunion_ci'         ,
    'reunion_rmw'        ,
    'reunion_r34'        ,
    'reunion_r50'        ,
    'reunion_r64'        ,
    'bom_lat'            ,
    'bom_lon'            ,
    'bom_type'           ,
    'bom_wind'           ,
    'bom_pres'           ,
    'bom_tnum'           ,
    'bom_ci'             ,
    'bom_rmw'            ,
    'bom_r34'            ,
    'bom_r50'            ,
    'bom_r64'            ,
    'bom_roci'           ,
    'bom_poci'           ,
    'bom_eye'            ,
    'bom_pos_method'     ,
    'bom_pres_method'    ,
    'nadi_lat'           ,
    'nadi_lon'           ,
    'nadi_cat'           ,
    'nadi_wind'          ,
    'nadi_pres'          ,
    'wellington_lat'     ,
    'wellington_lon'     ,
    'wellington_wind'    ,
    'wellington_pres'    ,
    'ds824_lat'          ,
    'ds824_lon'          ,
    'ds824_stage'        ,
    'ds824_wind'         ,
    'ds824_pres'         ,
    'td9636_lat'         ,
    'td9636_lon'         ,
    'td9636_stage'       ,
    'td9636_wind'        ,
    'td9636_pres'        ,
    'td9635_lat'         ,
    'td9635_lon'         ,
    'td9635_wind'        ,
    'td9635_pres'        ,
    'td9635_roci'        ,
    'neumann_lat'        ,
    'neumann_lon'        ,
    'neumann_class'      ,
    'neumann_wind'       ,
    'neumann_pres'       ,
    'mlc_lat'            ,
    'mlc_lon'            ,
    'mlc_class'          ,
    'mlc_wind'           ,
    'mlc_pres'           ,
    'bom_gust'           ,
    'bom_gust_per'       ,
    'reunion_gust'       ,
    'reunion_gust_per'   ,
    'dist2land'                   ,
    'landfall'                   ,
    'usa_r34'                   ,
    'usa_r50'                   ,
    'usa_r64'                   ,
    'usa_sshs'                   ,
    'usa_poci'                   ,
    'usa_roci'                   ,
    'usa_rmw'                   ,
    'usa_eye'                   ,
    'usa_gust'                   ,
    'usa_seahgt'                   ,
    'usa_searad'                   ,
    'storm_speed'                   ,
    'storm_dir'                   ,
    'nature'                   ,
    #'wmo_wind'                   ,
    #'wmo_agency'                   ,
    #'wmo_pres'                   ,
    'track_type'                   ,
    'main_track_sid'                   ,
    'iflag'                   ,
    'basin'                   ,
    'subbasin'                   ,
    'iso_time'                   ,
    'usa_atcf_id'                   ,
    'usa_record'                   ]


#def process_ibtracs(ibtracs_file):
    

def saffir_simpson(wsp, units):
    if units == 'm/s':
        if wsp < 33:
            return 'Tropical Storm'
        elif 33 <= wsp <= 42:
            return 'Category 1'
        elif 42 < wsp <= 49:
            return 'Category 2'
        elif 49 < wsp <= 57:
            return 'Category 3'
        elif 57 < wsp <= 70:
            return 'Category 4'
        elif wsp > 70:
            return 'Category 5'
    elif units == 'knots' or units == 'kts':
        if wsp < 64:
            return 'Tropical Storm'
        elif 64 <= wsp <= 82:
            return 'Category 1'
        elif 82 < wsp <= 95:
            return 'Category 2'
        elif 95 < wsp <= 112:
            return 'Category 3'
        elif 112 < wsp <= 136:
            return 'Category 4'
        elif wsp > 136:
            return 'Category 5'
    elif units == 'mph':
        if wsp < 74:
            return 'Tropical Storm'
        elif 74 <= wsp <= 95:
            return 'Category 1'
        elif 95 < wsp <= 110:
            return 'Category 2'
        elif 110 < wsp <= 129:
            return 'Category 3'
        elif 129 < wsp < 157:
            return 'Category 4'
        elif wsp >= 157:
            return 'Category 5'
    else:
        raise ValueError("Wind speed units must be 'm/s', 'kts', or 'mph'.")
    
def read_tempest(file, colnames):
    
    # Reads in file
    df = pd.read_csv(file, sep='\s+', names=colnames)
    
    # Takes individual year, month, day, hour columns and transforms to pd.date_time column
    df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']], errors='coerce')
    
    # Drops year, month, day, hour columns
    df = df[['time', 'lon_x', 'lat_y', 'lon', 'lat', 'slp', 'wsp', 'sfc_phi']]
    
    # Selects indices where a new track starts, assigns to array
    run_idx = df[df.lon_x=='start'].index.tolist()
    
    # Quantifies the max number of timesteps for all tracks
    max_tsteps = df['lat_y'].iloc[run_idx].values.max()

    # Separates dataframe into individual dataframes, split on new tracks
    dfs = [df.iloc[run_idx[n]+1:run_idx[n+1]] for n in range(len(run_idx)-1)]
    
    # Drops lon_x and lat_y columns
    dfs = [dfi[['time', 'lon', 'lat', 'slp', 'wsp', 'sfc_phi']] for dfi in dfs]
    
    # Resets index from previous dataframe splits
    dfs = [dfi.reset_index(drop=True) for dfi in dfs]
    
    # Pads dataframes with NaN rows so they're all the same length as the longest dataframe
    dfs = [dfi.reindex(range((max_tsteps-len(dfi))+len(dfi))) for dfi in dfs]
    
    # Converts longitude from [0, 360] to [-180, 180] 
    #for dfi in dfs:
    #    dfi['lon'] = dfi['lon'].map(lambda x: np.mod((x+180), 360)-180)
        
    # Makes sure lat, lon, wsp, slp, phi columns are floats and not str
    #for dfi
    
    return dfs