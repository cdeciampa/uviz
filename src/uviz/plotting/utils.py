import numpy as np

from haversine import inverse_haversine, Direction, Unit

import bokeh.palettes
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class CustomColorbars():
    """
    Class to store my custom outgoing longwave radiation and recipitation colormaps.
    """
    def __init__(self, target_cmap):
        
        self.out_cmap, self.out_levels, self.norm = self.get_cmap(target_cmap)
        
    def get_cmap(self, target_cmap):
        
        if target_cmap == 'FLUT':
            brightness_temps = np.array([-110, -92.1, -92, -80, -70, -60, -50, -42, 
                                         -30, -29.9, -20, -10, 0, 10, 20, 30, 40, 57])
            levels = np.array(list(map(self.T_to_FLUT, brightness_temps)))
            fracs = levels-self.T_to_FLUT(-110, 'C')
            fracs = fracs/fracs[-1]

            flut_colors = ['#ffffff', '#ffffff', '#e6e6e6', '#000000', 
                           '#ff1a00', '#e6ff01', '#00e30e', '#010073', 
                           '#00ffff', '#bebebe', '#acacac', '#999999', 
                           '#7e7e7e', '#6c6c6c', '#525252', '#404040', 
                           '#262626', '#070707']
            colormap = mcolors.LinearSegmentedColormap.from_list('FLUT CIMSS', list(zip(fracs, flut_colors)), N=1200)
            norm = mcolors.Normalize(vmin=levels[0], vmax=levels[-1])
        
        elif target_cmap == 'nws_precip':
            nws_precip_colors = [
                "#ffffff",  # 0.00 - 0.01 inches  white
                "#4bd2f7",  # 0.01 - 0.10 inches  light blue
                "#699fd0",  # 0.10 - 0.25 inches  mid blue
                "#3c4bac",  # 0.25 - 0.50 inches  dark blue
                "#3cf74b",  # 0.50 - 1.00 inches  light green
                "#3cb447",  # 1.00 - 1.50 inches  mid green
                "#3c8743",  # 1.50 - 2.00 inches  dark green
                "#1f4723",  # 2.00 - 3.00 inches  darkest green
                "#f7f73c",  # 3.00 - 4.00 inches  yellow
                "#fbde88",  # 4.00 - 5.00 inches  weird tan
                "#f7ac3c",  # 5.00 - 6.00 inches  orange
                "#c47908",  # 6.00 - 8.00 inches  dark orange
                "#f73c3c",  # 8.00 - 10.00 inch  bright red
                "#bf3c3c",  # 10.00 - 15.00 inch  mid red
                "#6e2b2b",  # 15.00 - 20.00 inch  dark red
                "#f73cf7",  # 20.00 - 25.00 inch  bright pink
                "#9974e4",  # 25.00 - 30.00 inch  purple
                #"#404040",  # 30.00 - 40.00 inch  dark gray because of mpl
                "#c2c2c2"  # 30.00 - 40.00 inch  gray
                ]
            colormap = mcolors.ListedColormap(nws_precip_colors, 'nws_precip')
            levels = [0.0, 0.01, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 
                      4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
            #levels = [0.01, 0.10, 0.25, 0.50, 1.0, 1.5, 2.0, 3.0, 
            #4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
            norm = mcolors.BoundaryNorm(boundaries=levels, ncolors=len(levels))

        else:
            raise ValueError("Must choose either 'FLUT' or 'nws_precip' colormap.")
        
        return colormap, levels, norm

    def T_to_FLUT(self, T, unit='C'):
        """
        Converts brightness temperature to outgoing longwave 
        radiation flux using the Stefan-Boltzmann Law. 
        Used to make pseudo-enhanced satellite plots.
        
        Parameters:
        ----------------
        T    :: float, temperature
        unit :: str (optional), unit of temperature. Default is 'C'.
        """
        
        if unit == 'C':
            T += 273.15
        sigma = 5.6693E-8
        olr = sigma*(T**4)

        return olr
    
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
    diverge_cmap = bokeh.palettes.diverging_palette(palette[:palette_cutoff], palette_split[::-1], n=ncolors, midpoint=diverge_point_norm)
    
    return diverge_cmap

class SaffirSimpson():
    def __init__(self, var, units):
        self.units = units
        self.var = var
        wsp_units = ['m/s', 'knots', 'kts', 'mph']
        mslp_units = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals', 
                      'mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']
        
        if self.units in wsp_units:
            self.category = self.sshs_wsp(self.var, self.units)
        elif self.units in mslp_units:
            self.category = self.sshs_mslp(self.var, self.units)
            
        self.color = self.sshs_color()
        
    def sshs_wsp(self, wsp, units):
        """
        Takes wind speed, converts to mph, returns corresponding category from SSHWS
        Parameters:
        -------------
        wsp : ndarray or variable
        units : str
        
        Returns: category : ndarray
        """
        # Converts units to mph so I only needed to write one function.
        if units == 'm/s':
            wsp = wsp * 2.2369
        elif units == 'knots' or units == 'kts':
            wsp = wsp * 1.15077945
        elif units == 'mph':
            wsp = wsp
        else:
            raise ValueError("Wind speed units must be 'm/s', 'kts', or 'mph'.")
            
        # Categorizes wind speed
        category = np.vectorize(lambda x: 'Category 5' if x >= 157.0 
                                else ('Category 4' if np.logical_and(x < 157.0, x > 129.0) 
                                else ('Category 3' if np.logical_and(x > 110.0, x <= 129.0) 
                                else ('Category 2' if np.logical_and(x > 95.0, x <= 110.0) 
                                else ('Category 1' if np.logical_and(x >= 74.0, x <= 95.0) 
                                else ('Tropical Storm' if np.logical_and(x > 38.0, x < 74.0) 
                                else 'Tropical Depression'))))))(wsp)
        
        return category
        

    def sshs_mslp(self, slp, unit):
        """
        Takes minimum sea level pressure, converts to pascals, returns corresponding category from SSHS.
        (According to Klotzbach et. al., 2020, modified for TS and TD designation using Kantha, 2006)
        Parameters:
        -------------
        wsp : ndarray or variable
        units : str
        """
        pascals = ['pascal', 'Pascal', 'pa', 'Pa', 'pascals', 'Pascals']
        hPa = ['mb', 'millibars', 'hPa', 'hectopascals', 'Hectopascals']

        if unit in pascals:
            slp = slp/100

        category = np.vectorize(lambda x: 'Category 5' if x <= 925.0 
                                else ('Category 4' if np.logical_and(x <= 946.0, x > 925.0) 
                                else ('Category 3' if np.logical_and(x <= 960.0, x > 946.0) 
                                else ('Category 2' if np.logical_and(x <= 975.0, x > 960.0) 
                                else ('Category 1' if np.logical_and(x <= 990.0, x > 975.0) 
                                else ('Tropical Storm' if np.logical_and(x < 1000.0, x > 990.0) 
                                else 'Tropical Depression'))))))(slp)
        return category

    def sshs_color(self):
        """
        Takes category on the Saffir Simpson Hurricane Scale, spits out correct color for plotting.
        """
        color = np.vectorize(lambda x: '#5EBAFF' if x == 'Tropical Depression'
                             else ('#00FAF4' if x == 'Tropical Storm'
                            else ('#FFF795' if x == 'Category 1' 
                            else ('#FFD821' if x == 'Category 2' 
                            else ('#FF8F20' if x == 'Category 3' 
                            else ('#FF6060' if x == 'Category 4' 
                            else '#C464D9' if x == 'Category 5' 
                            else ''))))))(self.category)
        return color

def basin_bboxes(basin_name='', lon_range=360, **kwargs):
    """
    Creates predefined bounding boxes if supplied correct name.
    
    Parameters:
    -------------------
    basin_name :: str, target basin name.
    lon_range  :: int (optional), defines maximum range of longitude. 
                        Options are 360 or 180, default is 360.
    
    Optional kwargs:
    -------------------
    custom_basin :: dict, options include {'center_coords', 'distance', 'unit'}, or
                            {'west_coord', 'east_coord', 'north_coord', 'south_coord'}.
    """
    
    basin_name = basin_name.lower()
    basins = ['north atlantic', 'north atlantic zoomed', 'florida', 'south florida', 'miami', 
              'south atlantic', 'east pacific', 'west pacific', 'north indian', 'south indian', 
              'australia', 'south pacific', 'conus', 'east conus']
    
    custom_basin = kwargs.get('custom_basin', {})
    
    if basin_name not in basins and not custom_basin:
        err = f"'{basin_name}' not in list of basins. Choose from {basins} or supply `custom_basin` dict."
        raise ValueError(err)
        
    if basin_name == 'miami':
        custom_basin['center_coords'] = (25.775163, -80.208615)
        
    # Option 1: supply a name for one of the predetermined bounding boxes.
    if basin_name == 'north atlantic':
        west_coord = -105.0
        east_coord = -5.0
        north_coord = 70.0
        south_coord = 0.0
    elif basin_name == 'north atlantic zoomed':
        west_coord = -100.0
        east_coord = -15.0
        north_coord = 50.0
        south_coord = 7.5
    elif basin_name == 'florida':
        west_coord = -90.0
        east_coord = -72.5
        north_coord = 32.5
        south_coord = 20.0
    elif basin_name == 'south florida':
        west_coord = -84.0
        east_coord = -79.0
        north_coord = 30.0
        south_coord = 24.0
    elif basin_name == 'south atlantic':
        west_coord = -105.0
        east_coord = -5.0
        north_coord = 65.0
        south_coord = 0.0
    elif basin_name == 'east pacific':
        west_coord = -180.0
        east_coord = -80.0
        north_coord = 65.0
        south_coord = 0.0    
    elif basin_name == 'west pacific':
        west_coord = 90.0
        east_coord = 180.0
        north_coord = 58.0
        south_coord = 0.0        
    elif basin_name == 'north indian':
        west_coord = 35.0
        east_coord = 110.0
        north_coord = 42.0
        south_coord = -5.0    
    elif basin_name == 'south indian':
        west_coord = 20.0
        east_coord = 110.0
        north_coord = 5.0
        south_coord = -50.0        
    elif basin_name == 'australia':
        west_coord = 90.0
        east_coord = 180.0
        north_coord = 0.0
        south_coord = -60.0        
    elif basin_name == 'south pacific':
        west_coord = 140.0
        east_coord = -120.0#+360
        north_coord = 0.0
        south_coord = -65.0        
    elif basin_name == 'conus':
        west_coord = -130.0
        east_coord = -65.0
        north_coord = 50.0
        south_coord = 20.0        
    elif basin_name == 'east conus':
        west_coord = -105.0
        east_coord = -60.0
        north_coord = 48.0
        south_coord = 20.0
    elif custom_basin:
        set1 = {'center_coords', 'distance', 'unit'}
        set2 = {'west_coord', 'east_coord', 'north_coord', 'south_coord'}
        
        # Option 2: supply a center point and a distance around it to return a bounding box.
        if set1 <= custom_basin.keys():
            center_pt = custom_basin['center_coords']
            dist = custom_basin['distance']
            acceptable_units = [item.value for item in Unit]
            
            if custom_basin['unit'] not in acceptable_units:
                raise ValueError(f"Acceptable units are: {acceptable_units}.")
            else:
                unit = custom_basin['unit']
                
            west_coord = inverse_haversine(center_pt, dist, Direction.WEST, unit=unit)[1]
            east_coord = inverse_haversine(center_pt, dist, Direction.EAST, unit=unit)[1]
            north_coord = inverse_haversine(center_pt, dist, Direction.NORTH, unit=unit)[0]
            south_coord = inverse_haversine(center_pt, dist, Direction.SOUTH, unit=unit)[0]
            
        # Option 2: supply a user-defined bounding box.
        elif set2 <= custom_basin.keys():
            west_coord = custom_basin['west_coord']
            east_coord = custom_basin['east_coord']
            north_coord = custom_basin['north_coord']
            south_coord = custom_basin['south_coord']
        
        # There are no other custom options.
        else:
            err = f"Must supply either all of {str(set1)} or all of {str(set2)} in `custom_basin` keys."
            raise KeyError(err)

    # There are no other options.
    else:
        raise ValueError("Supplied name not in given list.")
    
    # Converts longitude from [-180, 180] to [0, 360].
    if lon_range == 360:
        west_coord = np.mod(west_coord, 360)
        east_coord = np.mod(east_coord, 360)
        
    # Currently doesn't support radians.
    elif lon_range != 360 and lon_range != 180:
        raise ValueError("Acceptable values for `lon_range` are 180 or 360.")
        
    # Packs longitude range and latitude range into lists and returns them.
    lons = (west_coord, east_coord)
    lats = (south_coord, north_coord)
    
    return lons, lats
