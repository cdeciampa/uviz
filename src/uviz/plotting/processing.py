import os
import glob
import multiprocessing as mp

import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

class ModelData():
    def __init__(self, root_dir=None, parallel=False, regridded=False):
        """
        Class to read in CAM5-SE and CAM5-MPAS netCDF output files into
        a nested dictionary for easier analysis. Automatically pairs grid
        and landmask files to their appropriate data files.
        
        Parameters
        --------------
        root_dir  :: str, default='/gpfs/group/cmz5202/default/cnd5285';
                     root folder where data files, grid files, and landmask 
                     files are stored. Not to be confused with folder where 
                     just the data files are stored.
        parallel  :: bool, default=False. Whether or not to parallelize file
                     read-in using dask.
        regridded :: bool, default=False. Whether or not to read in regridded
                     files in addition to unstructured/native files. Throws
                     error because not all portions of code support identical
                     analysis with regridded files.
                     
        Returns
        --------------
        ModelData().storms :: nested dictionary
        """

        # Throws soft error if regridded=True
        if regridded == True:
            print('Use of regridded files is not supported for all analysis, \
                setting regridded=False instead.')
            regridded = False
        
        # Sets default if tracks_dir not supplied
        if not root_dir:
            root_dir = r"/gpfs/group/cmz5202/default/cnd5285"
        else:
            if not isinstance(root_dir, str):
                raise TypeError('Must supply `root_dir` as string.')
        
        # Declares list of storm file paths
        tracks_dir = os.path.join(root_dir, 'synth_events')
        track_folders = glob.glob(os.path.join(tracks_dir, '*storm*'))
        
        # Declares empty dictionary for self.storms
        self.storms = {}
        
        print('Opening data files.')
        for track in track_folders:
            self.read_files(track, parallel, regridded)
            
        print('Opening grid files.')
        # Declares MPAS file location, assigns to xarray Dataset
        c_mesh = os.path.join(root_dir, 'MPAS_3km/x20.835586.florida.init.CAM.nc')
        c_mesh_ds = xr.open_dataset(c_mesh)
            
        # Declares various CAM file locations, assigns to xarray Datasets
        p_mesh_root = os.path.join(root_dir, 'maps_and_grids')
        p_mesh_EXT = os.path.join(p_mesh_root, "ne0np4natlanticext.ne30x4.g_scrip.nc")
        p_mesh_REF = os.path.join(p_mesh_root, "ne0np4natlanticref.ne30x4.g_scrip.nc")
        p_mesh_WAT = os.path.join(p_mesh_root, "ne0np4natlanticwat.ne30x4.g_scrip.nc")
        
        p_mesh_ds_EXT = xr.open_dataset(p_mesh_EXT)
        p_mesh_ds_REF = xr.open_dataset(p_mesh_REF)
        p_mesh_ds_WAT = xr.open_dataset(p_mesh_WAT)
        
        print('Opening landmask files.')
        # Declares file location
        landmask_dir = os.path.join(root_dir, 'landmasks')
        
        # Declares MPAS file location, assigns to xarray Dataset
        c_land = os.path.join(landmask_dir, 'MPAS.VR3_landmask.nc')
        c_land_ds = xr.open_dataset(c_land)
        
        # Declares various CAM file locations, assigns to xarray Datasets
        p_land_EXT = os.path.join(landmask_dir, 'SE.VR28.NATL.EXT_landmask.nc')
        p_land_REF = os.path.join(landmask_dir, 'SE.VR28.NATL.REF_landmask.nc')
        p_land_WAT = os.path.join(landmask_dir, 'SE.VR28.NATL.WAT_landmask.nc')
        
        p_land_ds_EXT = xr.open_dataset(p_land_EXT)
        p_land_ds_REF = xr.open_dataset(p_land_REF)
        p_land_ds_WAT = xr.open_dataset(p_land_WAT)
            
        # Matches appropriate CAM grid and landmask files to data files
        print('Matching grid files and landmask files with data files.')
        for storm, key in self.storms.items():
            if len(key['h3pn_ds'].ncol) == 155981:
                self.storms[storm].update({'parent_grid': p_mesh_ds_EXT})
                self.storms[storm].update({'parent_landmask': p_land_ds_EXT})
            elif len(key['h3pn_ds'].ncol) == 119603:
                self.storms[storm].update({'parent_grid': p_mesh_ds_REF})
                self.storms[storm].update({'parent_landmask': p_land_ds_REF})
            elif len(key['h3pn_ds'].ncol) == 69653:
                self.storms[storm].update({'parent_grid': p_mesh_ds_WAT})
                self.storms[storm].update({'parent_landmask': p_land_ds_WAT})
            
            # Appends MPAS grid and landmask to self.storms dict
            self.storms[storm].update({'child_grid': c_mesh_ds})
            self.storms[storm].update({'child_landmask': c_land_ds})
                
    def read_files(self, track, parallel, regridded):
        """
        Reads in netCDF files
        """
        
        # Declares mesh file location
        parent_folder = os.path.join(track, '28km')
        child_folder = os.path.join(track, '3km')
        
        # Declares data file locations
        # Original exclusionary pattern [!nc_remap] doesn't work for Ian and Irma
        h2pn_file = os.path.join(parent_folder, '*h2*[!*_re*p].nc')
        h3pn_file = os.path.join(parent_folder, '*h3*[!*_re*p].nc')
        h4pn_file = os.path.join(parent_folder, '*h4*[!*_re*p].nc')
            
        h2cn_file = os.path.join(child_folder, '*h2*[!*_re*p].nc')
        h3cn_file = os.path.join(child_folder, '*h3*[!*_re*p].nc')
        h4cn_file = os.path.join(child_folder, '*h4*[!*_re*p].nc')
                                        
        h2pr_file = os.path.join(parent_folder, '*h2*.nc_remap.nc')
        h3pr_file = os.path.join(parent_folder, '*h3*.nc_remap.nc')
        h4pr_file = os.path.join(parent_folder, '*h4*.nc_remap.nc')

        h2cr_file = glob.glob(os.path.join(child_folder, '*h2*.nc_remap.nc'))
        h3cr_file = glob.glob(os.path.join(child_folder, '*h3*.nc_remap.nc'))
        h4cr_file = glob.glob(os.path.join(child_folder, '*h4*.nc_remap.nc'))
        
        # Opens data files, assigns to xarray datasets
        # Prevents PSU's HPC from erroring out with parallelization
        while True:
            try:
                h2pn_ds = xr.open_mfdataset(h2pn_file, parallel=parallel)
                h3pn_ds = xr.open_mfdataset(h3pn_file, parallel=parallel)
                h4pn_ds = xr.open_mfdataset(h4pn_file, parallel=parallel)
                
                h2cn_ds = xr.open_mfdataset(h2cn_file, parallel=parallel)
                h3cn_ds = xr.open_mfdataset(h3cn_file, parallel=parallel)
                h4cn_ds = xr.open_mfdataset(h4cn_file, parallel=parallel)

                if regridded == True:
                    h2pr_ds = xr.open_mfdataset(h2pr_file, parallel=parallel)
                    h3pr_ds = xr.open_mfdataset(h3pr_file, parallel=parallel)
                    h4pr_ds = xr.open_mfdataset(h4pr_file, parallel=parallel)

                    h2cr_ds = xr.open_mfdataset(h2cr_file, parallel=parallel)
                    h3cr_ds = xr.open_mfdataset(h3cr_file, parallel=parallel)
                    h4cr_ds = xr.open_mfdataset(h4cr_file, parallel=parallel)
                    
                else:
                    h2pr_ds = None
                    h3pr_ds = None
                    h4pr_ds = None
                    h2cr_ds = None
                    h3cr_ds = None
                    h4cr_ds = None
                break
                
            except OSError:
                print('Erroring out, trying again')
                continue
            except UnboundLocalError as e:
                print(e)
                break
        
        # Appends new variables from self.new_vars() to datasets
        self.new_vars(h3pn_ds, h4pn_ds)
        self.new_vars(h3cn_ds, h4cn_ds)
        
        if regridded == True:
            self.new_vars(h3pr_ds, h4pr_ds)
            self.new_vars(h3cr_ds, h4cr_ds)
        
        # Creates dictionary of datasets associated with each track
        track_ext = track.split('_')[-1]
        try:
            int(track_ext)
            track_name = f'storm_{track_ext}'
        except:
            track_name = track_ext
        attrs = {track_name:{
            'h2pn_ds':h2pn_ds, 'h3pn_ds':h3pn_ds, 'h4pn_ds':h4pn_ds, 
            'h2pr_ds':h2pr_ds, 'h3pr_ds':h3pr_ds, 'h4pr_ds':h4pr_ds, 
            'h2cn_ds':h2cn_ds, 'h3cn_ds':h3cn_ds, 'h4cn_ds':h4cn_ds, 
            'h2cr_ds':h2cr_ds, 'h3cr_ds':h3cr_ds, 'h4cr_ds':h4cr_ds}}

        # Appends above dictionary to self.storms
        self.storms.update(attrs)

    def new_vars(self, h3_ds=None, h4_ds=None):
        """
        Function to ensure each Dataset has new variables:
        1) wind speed at 850 mb level
        2) maximum 850 mb wind speed,
        3) maximum surface wind speed
        4) total precipitation 
        5) maximum hourly rate of precipitation
        
        Parameters
        ----------------
        h3_ds :: xr.Dataset, 3-hourly data that includes wind speeds
        h4_ds :: xr.Dataset, hourly data that includes precipitation
        """
        if not h3_ds and not h4_ds:
            raise ValueError('Must supply at least one of h3_ds or h4_ds.')

        if h3_ds:
            h3_ds['WSP850'] = mpcalc.wind_speed(h3_ds['U850'], h3_ds['V850']).metpy.convert_units('mph').metpy.dequantify()
            h3_ds['WSP850_MAX'] = (h3_ds['WSP850'].max(dim='time') * units('mph')).metpy.dequantify()
            h3_ds['U10_MAX'] = h3_ds['U10'].metpy.convert_units('mph').max(dim='time').metpy.dequantify()
            
            # Calculates bulk wind shear for 850-200mb level (vertical wind shear for TCs)
            h3_ds['SHEAR_TC'] = ((h3_ds['U200'] - h3_ds['U850'])**2 + (h3_ds['V200'] - h3_ds['V850'])**2)**(1/2)
            h3_ds['SHEAR_TC'] = (h3_ds['SHEAR_TC'] * 1.94384 * units('knots')).metpy.dequantify()
            
            # Calcuates CAPE

        if h4_ds:
            h4_ds['PRECT_TOT'] = (h4_ds['PRECT'].metpy.convert_units('in/hour').sum(dim='time') * units('hour')).metpy.dequantify()
            h4_ds['PRECT_MAX'] = (h4_ds['PRECT'].metpy.convert_units('in/hour').max(dim='time')).metpy.dequantify()
            h4_ds['PRECT_MAX_CONV'] = (h4_ds['PRECC'].metpy.convert_units('in/hour').max(dim='time')).metpy.dequantify()
