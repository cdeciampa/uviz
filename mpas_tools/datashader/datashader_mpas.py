#!/usr/bin/env conda run -n datashader_tools python

import os
import glob
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd

import dask.dataframe as dd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.tri import Triangulation
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

import holoviews as hv
from holoviews import opts

from holoviews.operation.datashader import rasterize as hds_rasterize
import geoviews.feature as gf # only needed for coastlines
from datashader.mpl_ext import dsshow, alpha_colormap

from numba import jit

import math as math
import geocat.datafiles as gdf  # Only for reading-in datasets
from xarray import open_mfdataset
import xarray as xr

import metpy.calc as mpcalc
from metpy.units import units

# This entire block helps with relative imports
import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
from database.utils import choose_forecast, choose_level

import argparse

hv.extension("bokeh","matplotlib")

opts.defaults(
    opts.Image(width=1200, height=600),
    opts.RGB(width=1200, height=600))

# This funtion splits a global mesh along longitude
# Examine the X coordinates of each triangle in 'tris'. 
# Return an array of 'tris' where only those triangles
# with legs whose length is less than 't' are returned.

def unzipMesh(x,tris,t):
    return tris[(np.abs((x[tris[:,0]])-(x[tris[:,1]])) < t) &\
                (np.abs((x[tris[:,0]])-(x[tris[:,2]])) < t)]

# Compute the signed area of a triangle

def triArea(x,y,tris):
    return ((x[tris[:,1]]-x[tris[:,0]]) *\
            (y[tris[:,2]]-y[tris[:,0]])) - ((x[tris[:,2]]-x[tris[:,0]]) *\
                                            (y[tris[:,1]]-y[tris[:,0]]))

# Reorder triangles as necessary so they all have counter clockwise winding order. 
# CCW is what Datashader and MPL require.

def orderCCW(x,y,tris):
    tris[triArea(x,y,tris)<0.0,:] = tris[triArea(x,y,tris)<0.0,::-1]
    return(tris)

# Create a Holoviews Triangle Mesh suitable for rendering with Datashader
# This function returns a Holoviews TriMesh that is created from a list of coordinates, 
# 'x' and 'y', an array of triangle indices that addressess the coordinates in 'x' and 'y', 
# and a data variable 'var'. The data variable's values will annotate the triangle vertices

def createHVTriMesh(x,y,triangle_indices, var,n_workers=1):
    # Declare verts array
    verts = np.column_stack([x, y, var])


    # Convert to pandas
    verts_df  = pd.DataFrame(verts,  columns=['x', 'y', 'z'])
    tris_df   = pd.DataFrame(triangle_indices, columns=['v0', 'v1', 'v2'])

    # Convert to dask
    verts_ddf = dd.from_pandas(verts_df, npartitions=n_workers)
    tris_ddf = dd.from_pandas(tris_df, npartitions=n_workers)

    # Declare HoloViews element
    tri_nodes = hv.Nodes(verts_ddf, ['x', 'y', 'index'], ['z'])
    trimesh = hv.TriMesh((tris_ddf, tri_nodes))
    return(trimesh)

# Triangulate MPAS primary mesh:
# Triangulate each polygon in a heterogenous mesh of n-gons by connecting
# each internal polygon vertex to the first vertex. Uses the MPAS
# auxilliary variables verticesOnCell, and nEdgesOnCell.
# The function is decorated with Numba's just-in-time compiler so that it is translated into
# optimized machine code for better peformance

@jit(nopython=True)
def triangulatePoly(verticesOnCell, nEdgesOnCell):

    # Calculate the number of triangles. nEdgesOnCell gives the number of vertices for each 
    # cell (polygon)
    # The number of triangles per polygon is the number of vertices minus 2.
    
    nTriangles = np.sum(nEdgesOnCell - 2)

    triangles = np.ones((nTriangles, 3), dtype=np.int64)
    nCells = verticesOnCell.shape[0]
    triIndex = 0
    for j in range(nCells):
        for i in range(nEdgesOnCell[j]-2):
            triangles[triIndex][0] = verticesOnCell[j][0]
            triangles[triIndex][1] = verticesOnCell[j][i+1]
            triangles[triIndex][2] = verticesOnCell[j][i+2]
            triIndex += 1

    return triangles

def set_up_mesh(mesh_ds, n_workers=1):
    # Fetch lat and lon coordinates for the primal and dual mesh.
    lonCell = mesh_ds['lonCell'].values * 180.0 / math.pi
    latCell = mesh_ds['latCell'].values * 180.0 / math.pi
    lonCell = ((lonCell - 180.0) % 360.0) - 180.0

    lonVertex = mesh_ds['lonVertex'].values * 180.0 / math.pi
    latVertex = mesh_ds['latVertex'].values * 180.0 / math.pi
    lonVertex = ((lonVertex - 180.0) % 360.0) - 180.0

    # Get triangle indices for each vertex in the MPAS file. Note, indexing in MPAS starts from 1, not zero :-(
    tris = mesh_ds.cellsOnVertex.values - 1

    # Guarantees consistent clockwise winding order (required by Datashade and Matplotlib)
    tris = orderCCW(lonCell,latCell,tris)

    # Unzip the mesh along a constant line of longitude for PCS coordinates (central_longitude=0.0)
    central_longitude = 0.0
    projection = ccrs.Robinson(central_longitude=central_longitude)
    tris = unzipMesh(lonCell,tris,90.0)

    # Project verts from geographic to PCS coordinates
    xPCS, yPCS, _ = projection.transform_points(ccrs.PlateCarree(), lonCell, latCell).T
    
    return xPCS, yPCS, tris, n_workers

def datashader_wrapper(mesh_ds, unstructured_ds, primalVarName, time, level=None, 
                       pixel_height=400, pixel_width=400, pixel_ratio=1, x_sampling=None, y_sampling=None):
    
    # Selects target variable from dataset based on timestep
    primalVar = unstructured_ds[primalVarName].isel(time=time).values
    
    if np.ndim(primalVar) > 1 and level==None:
        raise ValueError('Select a level to knock this down to a 1D array.')
    elif np.ndim(primalVar) > 1 and level != None:
        primalVar = unstructured_ds[primalVarName].sel(lev=level, method='nearest').isel(time=time).values
    
    xPCS, yPCS, tris, n_workers = set_up_mesh(mesh_ds)
    
    trimesh = createHVTriMesh(xPCS,yPCS,tris, primalVar,n_workers=n_workers)
    
    # Use precompute so it caches the data internally
    rasterized = hds_rasterize(trimesh, aggregator='mean', precompute=True, height=pixel_height, width=pixel_width, 
                               pixel_ratio=pixel_ratio, x_sampling=x_sampling, y_sampling=y_sampling)
    
    return rasterized

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--init_date', type=str, default='0830')
parser.add_argument('-pr', '--pixel_ratio', type=int, default=10)
parser.add_argument('-pw', '--pixel_width', type=int, default=400)
parser.add_argument('-ph', '--pixel_height', type=int, default=400)
parser.add_argument('-cm', '--cmap', type=str, default='gist_yarg')

parser.add_argument('-ts', '--timestep', type=int, required=True)
parser.add_argument('-tv', '--target_variable', type=str, required=True)
parser.add_argument('-ti', '--title', type=str, required=True)
args = parser.parse_args()

print('Reading in files.')
hires_folder = r"/gpfs/group/cmz5202/default/cnd5285/MPAS_3km"
hires_files = glob.glob(os.path.join(hires_folder, '*.cam.h0.*.nc'), recursive=False)
if args.init_date == '0830':
    hires_init = [f for f in hires_files if '08-30' in f]
elif args.init_date == '0831':
    hires_init = [f for f in hires_files if '08-31' in f]
elif args.init_date == '0901':
    hires_init = [f for f in hires_files if '09-01' in f]
elif args.init_date == '0902':
    hires_init = [f for f in hires_files if '09-02' in f]
else:
    raise ValueError("Pick an initialization date of '0830','0831', '0901', or '0902'.")

hires_ds = xr.open_mfdataset(hires_init)
hires_mesh_file = os.path.join(hires_folder, 'x20.835586.florida.init.CAM.nc')
hires_mesh = xr.open_dataset(hires_mesh_file, decode_times=False)

print('Assigning variables from command line.')
target_var = args.target_variable
target_time = args.timestep
pixel_height = args.pixel_height
pixel_width = args.pixel_width
pixel_ratio = args.pixel_ratio
fig_title = args.title
cmap = args.cmap

hv.output(dpi=200, backend='bokeh', fig='png')
central_longitude = 0.0
projection = ccrs.Robinson(central_longitude=central_longitude)

print(f'Rasterizing with {pixel_ratio} pixel ratio.')
rasterized = datashader_wrapper(hires_mesh, hires_ds, target_var, target_time, pixel_ratio=pixel_ratio, pixel_height=pixel_height, pixel_width=pixel_width)
final_img = rasterized.opts(tools=['hover'], colorbar=True, cmap=cmap, title=fig_title) * gf.coastline(projection=projection)

print('Exporting image.')
hv.save(final_img, f'./output_imgs/{fig_title}.png', backend='bokeh')

end = time.time()
print(f'Time elapsed: {end-start} seconds.')