# Heavily borrowed from/built on the work of Phillip Chmielowiec: https://github.com/NCAR/geocat-scratch/tree/main/polymesh

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import shapely
import spatialpandas as sp
import pyarrow as pa

import bokeh.palettes
from bokeh.io import export_svg

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import rasterize as hds_rasterize

import geoviews.feature as gf
import logging

# Hides inconsequential warning about renaming x and y dims to x and y
logging.getLogger("UserWarning").setLevel(logging.CRITICAL)

class Polymesh():
    def __init__(self, mesh_ds, data_ds=None, model='mpas', projection=ccrs.PlateCarree()):
        """
        Given an xarray dataset, constructs a polygon mesh of the native grid
        suitable for plotting with Datashader.
        Parameters
        ---------------
        mesh_ds : xarray dataset of the unstructured mesh file
        data_ds : xarray dataset of the unstructured data file
        model : model name (limited to 'mpas' or 'cam')
        projection : ccrs., Cartopy projection for coordinate projection/transform.
        """
        
        # Assigns variables from input
        self.mesh_ds = mesh_ds
        self.data_ds = data_ds
        self.model = model
        self.projection = projection

        # Node (vertex) indices
        self.face_nodes = self.fix_face_nodes()
        self.n_faces, self.n_face_nodes = self.face_nodes.shape
        
        # Original x and y node (vertex) coordinates (converted from [0, 360] to [-180, 180])
        if self.model == 'mpas':
            face_node_x = np.mod(np.rad2deg(self.mesh_ds.lonVertex.values) - 180.0, 360.0) - 180.0
            face_node_y = np.rad2deg(self.mesh_ds.latVertex.values)
        elif self.model =='cam':
            face_node_x = np.mod(self.mesh_ds.grid_corner_lon.values - 180.0, 360.0) - 180.0
            face_node_y = self.mesh_ds.grid_corner_lat.values
        
        # Creates polygons given polygon node indices
        self.polygon_array, self.cyclic_index = self.create_polygon_array(face_node_x, face_node_y)
        
        # Creates a SpatialPandas GeoDataFrame from polygon mesh information.
        self.gdf = self.construct_mesh()
        
    def fix_face_nodes(self):
        """
        Given MPAS's icosahedral mesh, with some 4-sided, 5-sided, and 6-sided polygons, returns 
        a corrected array of indices where polygons with sides n < 7 have final indices == initial
        indices (instead of 0).
        """
        
        if self.model == 'mpas':
            # MPAS indexing starts at 1, not 0 (any original 0 value ~ NaN)
            face_nodes = self.mesh_ds.verticesOnCell.values - 1
            
            # Replaces all -1 values with first value in array to force 7-sided shapes
            face_nodes[:, 4] = np.where(face_nodes[:, 4] == -1, face_nodes[:, 0], face_nodes[:, 4])
            face_nodes[:, 5] = np.where(face_nodes[:, 5] == -1, face_nodes[:, 0], face_nodes[:, 5])
            face_nodes[:, 6] = np.where(face_nodes[:, 6] == -1, face_nodes[:, 0], face_nodes[:, 6])
            
            # Ensures face nodes are returned as integers
            face_nodes = face_nodes.astype(int)
            
        elif self.model == 'cam':
            # CAM5-SE doesn't have given node indices and I didn't feel like making an ndarray of true indices
            face_nodes = np.indices(self.mesh_ds.grid_corner_lat.values.shape)[0]
        
        return face_nodes
    
    def find_cyclic_polygons(self, poly_data):
        """ 
        For polygons that cross +/- 180 deg longitude, those need to be split in two
        to avoid 'improper rendering'.
        Parameters:
        --------------
        poly_data : ndarray of polygons
        """
        
        # Finds any polygon that contains both + and - lon values (ie. crosses +/- 180 or 0 deg)
        x = poly_data[:, 0::2]
        out_left = np.all((-x - np.abs(x)) == 0, axis=1)
        out_right = np.all((x - np.abs(x)) == 0, axis=1)
        out = out_left | out_right
        
        # Creates index of cyclic cells to be dropped
        cc_index = np.arange(0, self.n_faces, 1)
        cc_index = cc_index[~out]
        
        # Find all polygons within some +/- degrees of center lon
        center_buffer = 80
        cc_polys = poly_data[cc_index]
        corrected_index = np.any(np.abs(cc_polys[:, ::2]) < center_buffer, axis=1)
        
        # Update cyclic cells to exclude center polygons
        cc_index = cc_index[~corrected_index]
        
        return cc_index
    
    def ensure_ccw_polygons(self, poly_list):
        """
        Function to ensure polygons are in CCW winding order for proper plotting in Datashader.
        Parameters
        --------------
        poly_list : ndarray of polygons
        """
        
        # Creates integer index of polygons for easy indexing
        poly_idx = np.arange(0, len(poly_list))

        # Finds vertex index with lowest x value, then highest y value (if there's a tie) - ie bottom-right/southeast vertex
        # This assumes a simple polygon (one that does not intersect itself and has no holes)
        vert_idx = np.lexsort([-poly_list[:, :, 1], poly_list[:, :, 0]])[:, 0]
        
        # If target idx is at the end of the array, convert it to -1 for easy indexing
        vert_idx[vert_idx == poly_list.shape[1]-1] = -1

        # Finds edges on either side of target vertex
        edges1 = np.stack([poly_list[poly_idx, vert_idx-1, :], 
                           poly_list[poly_idx, vert_idx, :]], 1) 
        edges2 = np.stack([poly_list[poly_idx, vert_idx, :], 
                           poly_list[poly_idx, vert_idx+1, :]], 1)

        # Take the sum of the cross product of the edge proceeding the vertex and the edge succeeding the vertex
        ccw = np.sum(np.cross(edges1, edges2), axis=1)

        # Finds CW polygons and reverses them in-place
        poly_list[ccw<0.0,:] = poly_list[ccw<0.0,::-1]

        return poly_list
        
    def create_polygon_array(self, x, y):
        """ Converts coordinate and face node data to
        a polygon array, taking into account cyclic
        polygons (those that cross the anti-meridian [+/- 180]).
        
        Parameters (from Class obj)
        ----------
        x : ndarray
            coordinate values for 'x' coordinates
        y : ndarray
            coordinate values for 'y' coordinates
            
        Returns
        -------
        polygon_array_ccw : ndarray
            Array containing CCW polygon coordinates (new)
        cyclic_index : array
            Array containing the index of the cyclic polygons
        """
        
        
        # Gets polygons' coordinate data
        if self.model =='mpas':
            x_coords = x[self.face_nodes]
            y_coords = y[self.face_nodes]
        elif self.model =='cam':
            x_coords = x
            y_coords = y
        
        # Packs into array with shape == (self.n_faces, self.n_face_nodes*2)
        poly_data = np.stack([x_coords, y_coords], 2).reshape(x_coords.shape[0], -1)
        cyclic_index = self.find_cyclic_polygons(poly_data)
        cyclic_polys = poly_data[cyclic_index]
        
        # If there aren't any cyclic polygons:
        if len(cyclic_polys) == 0:
            polygon_array = poly_data.reshape(self.n_faces, self.n_face_nodes, 2)
            polygon_array_ccw = self.ensure_ccw_polygons(polygon_array)
            
            return polygon_array_ccw, None
        else:
            # Iterate over each cyclic polygon, splitting it into two
            poly_list = []
            new_poly_index = []
            for i, poly in enumerate(cyclic_polys):
                poly_left = poly.copy()
                poly_right = poly.copy()

                # Start in RHS
                if poly[0] > 0:
                    # Get Remaining x coordinates
                    x_remain_index = poly[2::2] > 0

                    # Update coordinates of Right Polygon
                    poly_right[2::2][~x_remain_index] = poly[2::2][~x_remain_index] + 360

                    # Update coordinates of Left Polygon
                    poly_left[0] = poly[0] - 360
                    poly_left[2::2][x_remain_index] = poly[2::2][x_remain_index] - 360

                # Start in LHS
                elif poly[0] < 0:
                    # Get Remaining x coordinates
                    x_remain_index = poly[2::2] < 0

                    # Update coordinates of Left Polygon
                    poly_left[2::2][~x_remain_index] = poly[2::2][~x_remain_index] - 360

                    # Update coordinates of Right Polygon
                    poly_right[0] = poly[0] + 360
                    poly_right[2::2][x_remain_index] = poly[2::2][x_remain_index] + 360

                # Ignore
                else:
                    # longitude = 0, might be the pole issue
                    print("Missing Polygon at Index: {}".format(i))
                    continue

                # Ensure longitude values are within +/- 180 bound
                poly_left[::2][poly_left[::2] < -180] = -180.0
                poly_right[::2][poly_right[::2] > 180] = 180.0

                # Store New Polygons and Corresponding Indicies
                poly_list.extend([poly_left, poly_right])
                #new_poly_index.extend([self.cyclic_index[i], self.cyclic_index[i]])

            new_poly_data = np.array(poly_list)
            # MAYBE new_poly_index = np.array(new_poly_index)
            
            # Projects to new coordinate system from PlateCarree()
            orig_poly_coords = self.projection.transform_points(ccrs.PlateCarree(), 
                                                                x_coords, y_coords)[:, :, :2]
            
            # Removes polygons that were split in half from original array to avoid double-counting
            orig_poly_coords = np.delete(orig_poly_coords, cyclic_index, axis=0)
            new_poly_coords = self.projection.transform_points(ccrs.PlateCarree(), 
                                                               new_poly_data[:, 0::2], 
                                                               new_poly_data[:, 1::2])[:, :, :2]
            polygon_array = np.vstack([orig_poly_coords, new_poly_coords])
            
            # Ensures returned polygons are in CCW winding order
            polygon_array_ccw = self.ensure_ccw_polygons(polygon_array)

            return polygon_array_ccw, cyclic_index
        
    def construct_mesh(self):
        """ Constructs a Polygon Mesh using the calculated
        polygon array and drop index for cyclic polygons
        
        Parameters (from Class obj)
        ----------
        polygon_array : ndarry
            Array containing Polygon Coordinates (original and new)
        drop_index : ndarray
            Array containing indices to cyclic polygons
            
        Returns
        -------
        gdf : GeoDataFrame
            Contains polygon geometry
        """
        
        # Create Shapely Polygon Object
        geo = shapely.polygons(self.polygon_array)

        # Get Coords and indicies for PyArrow
        arr_flat, part_indices = shapely.get_parts(geo, return_index=True)
        offsets1 = np.insert(np.bincount(part_indices).cumsum(), 0, 0)
        arr_flat2, ring_indices = shapely.get_rings(arr_flat, return_index=True)
        offsets2 = np.insert(np.bincount(ring_indices).cumsum(), 0, 0)
        coords, indices = shapely.get_coordinates(arr_flat2, return_index=True)
        offsets3 = np.insert(np.bincount(indices).cumsum(), 0, 0)
        coords_flat = coords.ravel()
        offsets3 *= 2

        # Create a PyArrow array with our Polygons
        _parr3 = pa.ListArray.from_arrays(pa.array(offsets3), pa.array(coords_flat))
        _parr2 = pa.ListArray.from_arrays(pa.array(offsets2), _parr3)
        parr = pa.ListArray.from_arrays(pa.array(offsets1), _parr2)

        # Create Spatial Pandas Polygon Objects from PyArrow
        polygons = sp.geometry.MultiPolygonArray(parr)

        # Store our Polygon Geometry in a GeoDataFrame
        return sp.GeoDataFrame({'geometry': polygons})
    
    def calc_derived(self, u, v, target_var='vorticity'):
        """
        Based off of a modified Greene's Theorem from Helms and Hart, 2013 
        (https://doi.org/10.1175/JAMC-D-12-0248.1)
        """

        if self.model != 'mpas':
            raise ValueError('Currently only supports MPAS.')

        acceptable_vars = ['vorticity', 'divergence', 'convergence', 
                           'shearing deformation', 'stretching deformation']
        if target_var.lower() not in acceptable_vars:
            raise ValueError(f"Choices of target_var limited to {acceptable_vars}.")

        # Extracts the indices of the dual mesh (triangular) edges
        edge_idx = self.mesh_ds.edgesOnVertex.values - 1

        # Extracts the indices of the dual mesh (triangular) nodes
        triangle_idx = self.mesh_ds.cellsOnVertex.values - 1

        # Converts the angle of the cross product between a hexagonal edge and north to degrees
        # This angle == angle between north and triangle edge 
        angles = np.rad2deg(self.mesh_ds.angleEdge.values)

        # Extracts the lengths of dual mesh edges (in meters)
        edge_dist = self.mesh_ds.dcEdge.values

        # Extracts the areas of the dual mesh
        triangle_areas = self.mesh_ds.areaTriangle.values

        # Extracts Coriolis values at primal mesh vertices/dual mesh centroids
        coriolis = self.mesh_ds.fVertex.values

        # Breaks the edge lengths into their components
        delta_x = np.multiply(np.abs(edge_dist), np.cos(90 - angles))
        delta_y = np.multiply(np.abs(edge_dist), np.sin(90 - angles))

        # Matches u and v components to dual mesh
        triag_us = u[triangle_idx]
        triag_vs = v[triangle_idx]

        # Multiplies varies combinations of u, v, dx, and dy
        udx = np.multiply(triag_us, delta_x[edge_idx])
        udy = np.multiply(triag_us, delta_y[edge_idx])
        vdx = np.multiply(triag_vs, delta_x[edge_idx])
        vdy = np.multiply(triag_vs, delta_y[edge_idx])

        if target_var == 'vorticity':

            # Adds udx + vdy
            udx_vdy = np.sum([udx, vdy], axis=0)

            # Takes rowwise sum of udx + vdy
            vort_num = np.sum(udx_vdy, axis=1)

            # Divides by the area of the triangle to get relative vorticity at
            # triangle centroids/hexagon nodes
            vorticity = np.divide(vort_num, triangle_areas)

            # Adds in Coriolis to get absolute vorticity
            abs_vorticity = np.sum([vorticity, coriolis], axis=0)

            # Takes mean across hexagon nodes to get absolute vorticity at hexagon centroids
            hex_vort = np.mean(abs_vorticity[self.face_nodes], axis=1)

            return hex_vort

        elif target_var == 'divergence' or target_var == 'convergence':

            # Subtracts udy - vdx
            udy_vdx = np.subtract([udy, vdx], axis=0)

            # Takes rowwise sum of udy - vdx
            div_num = np.sum(udy_vdx, axis=1)

            # Divides by the area of the triangle to get divergence at 
            # triangle centroids/hexagon nodes
            divergence = np.divide(div_num, triangle_areas)

            # Takes mean across hexagon nodes to get divergence at hexagon centroids
            hex_div = np.mean(divergence[self.face_nodes], axis=1)

            return hex_div

        elif target_var == 'shearing deformation':

            # Subtracts vdy - udx
            vdy_udx = np.subtract([vdy, udx], axis=0)

            # Takes rowwise sum of vdy - udx
            shear_num = np.sum(vdy_udx, axis=1)

            # Divides by the area of the triangle to get shearing deformation at 
            # triangle centroids/hexagon nodes
            shearing = np.divide(shear_num, triangle_areas)

            # Takes mean across hexagon nodes to get shearing deformation at hexagon centroids
            hex_shear = np.mean(shearing[self.face_nodes], axis=1)

            return hex_shear

        elif target_var == 'stretching deformation':

            # Adds vdx + udy
            vdx_udy = np.sum([vdx, udy], axis=0)

            # Takes rowwise sum of vdx + udy
            stretch_num = np.sum(vdx_udy, axis=1)

            # Divides by the area of the triangle to get stretching deformation at 
            # triangle centroids/hexagon nodes
            stretching = np.divide(stretch_num, triangle_areas)

            # Takes mean across hexagon nodes to get stretching deformation at hexagon centroids
            hex_stretch = np.mean(stretching[self.face_nodes], axis=1)

            return hex_stretch
    
    def data_mesh(self, target_var, dims, fill='faces', **kwargs):
        """ Given a Variable Name and Dimensions, returns a
        GeoDataFrame containing geometry and fill values for
        the polygon mesh
        Parameters
        ----------
        target_var : string, required
            Name of data variable for rendering
        dims : dict, required
            Dictonary of dimensions for data variable (time, level, etc.)
        fill : string
            Method for calculating face values
            
        Optional kwargs
        -----------------
        derived_kw :: dict, only for visualizing derived quantities 
                        (vorticity, divergence, or deformation). 
                        Must include variable names for `u` and `v` in keys.
        
        Returns
        -------
        gdf : GeoDataFrame
            Contains polygon geometry and face values
        """
        
        # Ensure a valid variable name is passed through
        derived_opts = list(['vorticity', 'divergence', 'convergence', 
                            'shearing deformation', 'stretching deformation'])
        if target_var not in list(self.data_ds.data_vars) + derived_opts:
            raise ValueError('Chosen target variable not in DataSet.')
        
        # Unpacks option of plotting derived field
        derived_kw = kwargs.get('derived_kw', {})
        if target_var in derived_opts:
            if 'u' not in derived_kw.keys():
                raise ValueError("Must supply xarray variable name value for 'u' in 'derived_kw' kwarg.")
            elif 'v' not in derived_kw.keys():
                raise ValueError("Must supply xarray variable name value for 'v' in 'derived_kw' kwarg.")
                
        if fill == 'faces':
            if target_var not in derived_opts:
                face_array = self.data_ds[target_var].isel(dims).values
            elif target_var in derived_opts:
                u = self.data_ds[derived_kw['u']].isel(dims).values
                v = self.data_ds[derived_kw['v']].isel(dims).values
                face_array = self.calc_derived(u, v, target_var)
        elif fill == 'nodes':
            if self.model == 'mpas' or self.model == 'cam':
                raise ValueError(f"There aren't any data on the nodes for the {self.model} model.")
            else:
                face_array = self.mesh_ds[target_var].isel(dims).values[self.face_nodes].mean(axis=1)
        else:
            raise ValueError("Must supply 'faces' or 'nodes' for fill value.")
            
        # Raises an error if user supplies incorrect unstructured mesh for CAM
        if self.model == 'cam':
            if len(self.face_nodes) != len(face_array):
                raise ValueError("Must supply the correct CAM mesh.")
        
        # If there are cyclic polygons, append the split ones, then delete the original cyclic polys
        if len(self.face_nodes) < len(self.polygon_array):
            new_faces = np.repeat(face_array[self.cyclic_index], 2)
            face_array = np.hstack([face_array, new_faces])
            face_array = np.delete(face_array, self.cyclic_index)
        
        # Updates mesh polygon geodataframe with new chosen target_var
        self.gdf = self.gdf.assign(faces = face_array)
        
        return self.gdf

def plot_native(polymesh_df, proj=ccrs.PlateCarree(), plot_bbox=None, raster=True, save_fig=False, **kwargs):
    """
    Function to plot the native grid in either raster or vector format using holoviews/bokeh.
    Accepted file types are 'svg' for vector or 'png' for raster.
    
    Parameters:
    -------------------
    polymesh_df :: sp.gdf - the SpatialPandas GeoDataFrame output with polygons in one column and data in the other.
    
    proj        :: cartopy.ccrs, optional - coordinate projection system for plotting
    plot_bbox   :: list, optional - bounding box for plotting
    raster      :: bool, optional (default=True) - whether to plot raster (True) or vector (False) images.
    save_fig    :: bool, optional (default=False) - whether to output plot to a file.
    
    Optional kwargs:
    --------------------
    holoviews_kw   :: dict - expose plot options with `hv.help(hv.Polygons)`.
    datashader_kw  :: dict - expose rasterization options with `hv.help(hds_rasterize)`.
    coastline_kw   :: dict - expose plotting options with `hv.help(gf.coastline)`
    lakes_kw       :: dict - expose plotting options with `hv.help(gf.lakes)`.
    out_file_kw    :: dict - must supply filename, but expose other options with `hv.help(hv.save)` for
                        raster images or `?export_svg` for vector images.
    """
    
    # Retrieves kwargs
    holoviews_kw = kwargs.get('holoviews_kw', {})
    datashader_kw = kwargs.get('datashader_kw', {})
    coastline_kw = kwargs.get('coastline_kw', {})
    lakes_kw = kwargs.get('lakes_kw', {})
    out_file_kw = kwargs.get('out_file', {})
    
    if save_fig == True and not out_file_kw:
        raise ValueError('Must supply kwargs to out_file_kw if you wish to output your figure.')
    elif out_file_kw and 'filename' not in out_file_kw.keys():
        raise KeyError('Must supply filename to out_file_kw if you wish to output your figure.')
        
    if plot_bbox != None:
        lon_range, lat_range = plot_bbox
        x_range, y_range, _ = proj.transform_points(ccrs.PlateCarree(), 
                                                    np.array(lon_range), 
                                                    np.array(lat_range)).T
        if raster == True:
            datashader_kw['x_range'] = tuple(x_range)
            datashader_kw['y_range'] = tuple(y_range)
        elif raster == False:
            holoviews_kw['xlim'] = tuple(x_range)
            holoviews_kw['ylim'] = tuple(y_range)
    
    if raster == True:
        hv_polys = hv.Polygons(polymesh_df, vdims=['faces']).opts(color='faces') 
        rasterized = hds_rasterize(hv_polys, **datashader_kw) 
        out_plot = rasterized.opts(**holoviews_kw) 
        if coastline_kw:
            out_plot = out_plot * gf.coastline(projection=proj).opts(**coastline_kw)
        if lakes_kw:
            out_plot = out_plot * gf.lakes(projection=proj).opts(**lakes_kw)
        if save_fig == True:
            print('Exporting raster image...')
            hv.save(out_plot, center=False, **out_file_kw)
        else:
            return out_plot
    
    elif raster == False:
        out_plot = hv.Polygons(polymesh_df, vdims=['faces']).opts(**holoviews_kw) 
        if coastline_kw:
            out_plot = out_plot * gf.coastline(projection=proj).opts(**coastline_kw)
        if lakes_kw:
            out_plot = out_plot * gf.lakes(projection=proj).opts(**lakes_kw)
        if save_fig == True:
            svg_fig = hv.render(out_plot, backend='bokeh')
            svg_fig.output_backend = 'svg'
            svg_fig.background_fill_color = None
            svg_fig.border_fill_color = None
            print('Exporting vector image...')
            export_svg(svg_fig, filename=out_file_kw['filename'])
        else:
            return out_plot
        
def diverging_colormap(cmin, cmid, cmax, palette, ncolors=256):
    """
    Function to create a diverging colormap to use with Holoviews (which is not natively supported).
    Returns divergent colormap (one where the normalization above the midpoint != normalization 
    below the midpoint).
    
    Parameters:
    ----------------
    cmin     ::  float, int - minimum to clip plotting data (akin to mpl's vmin) for normalization
    cmid     ::  float, int - midpoint to begin divergence of colors
    cmax     ::  float, int - maximum to clip plotting data (akin to mpl's vmax) for normalization
    palette  ::  bokeh.palettes, list - original colormap (or list of colors) for diverging
    
    ncolors  :: optional, int (default=256), must match number of colors in the supplied palette.
    """

    diverge_point_norm = (cmid - cmin) / (cmax - cmin)
    palette_cutoff = round(diverge_point_norm * ncolors)
    palette_split = palette[palette_cutoff:]
    diverge_cmap = bokeh.palettes.diverging_palette(palette[:palette_cutoff], 
                                                    palette_split[::-1], n=ncolors, 
                                                    midpoint=diverge_point_norm)
    
    return diverge_cmap