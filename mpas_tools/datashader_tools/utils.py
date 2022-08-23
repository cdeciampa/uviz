import numpy as np
import pandas as pd
import dask.dataframe as dd
import cartopy.crs as ccrs
import xarray as xr
import math

import holoviews as hv
from holoviews.operation.datashader import rasterize as hds_rasterize

from numba import jit

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
    
    return xPCS, yPCS, tris, n_workers, projection

def datashader_wrapper(mesh_ds, unstructured_ds, primalVarName, time, level=None, 
                       pixel_height=400, pixel_width=400, pixel_ratio=1, x_sampling=None, 
                       y_sampling=None, lon_range=None, lat_range=None):
    
    # Selects target variable from dataset based on timestep
    primalVar = unstructured_ds[primalVarName].isel(time=time).values
    
    if np.ndim(primalVar) > 1 and level==None:
        raise ValueError('Select a level to knock this down to a 1D array.')
    elif np.ndim(primalVar) > 1 and level != None:
        primalVar = unstructured_ds[primalVarName].sel(lev=level, method='nearest').isel(time=time).values
    
    xPCS, yPCS, tris, n_workers, projection = set_up_mesh(mesh_ds)
    
    trimesh = createHVTriMesh(xPCS,yPCS,tris, primalVar,n_workers=n_workers)
    
    if lon_range != None and lat_range != None:
        x_range, y_range, _ = projection.transform_points(ccrs.PlateCarree(), np.array(lon_range), np.array(lat_range)).T
        x_range = tuple(x_range)
        y_range = tuple(y_range)
    else:
        x_range = None
        y_range = None
    
    # Use precompute so it caches the data internally
    rasterized = hds_rasterize(trimesh, aggregator='mean', precompute=True, height=pixel_height, 
                               width=pixel_width, pixel_ratio=pixel_ratio, x_sampling=x_sampling, 
                               y_sampling=y_sampling, x_range=x_range, y_range=y_range)
    
    return rasterized