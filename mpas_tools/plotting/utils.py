import matplotlib.colors as mcolors
import numpy as np

def nonlinear_colorbar(var_name=None):
    if var_name == 'PRECIP':
        colors = [
                '#ffffff',  # 0 inches
                "#04e9e7",  # 0.01 - 0.10 inches
                "#019ff4",  # 0.10 - 0.25 inches
                "#0300f4",  # 0.25 - 0.50 inches
                "#02fd02",  # 0.50 - 0.75 inches
                "#01c501",  # 0.75 - 1.00 inches
                "#008e00",  # 1.00 - 1.50 inches
                "#fdf802",  # 1.50 - 2.00 inches
                "#e5bc00",  # 2.00 - 2.50 inches
                "#fd9500",  # 2.50 - 3.00 inches
                "#fd0000",  # 3.00 - 4.00 inches
                "#d40000",  # 4.00 - 5.00 inches
                "#bc0000",  # 5.00 - 6.00 inches
                "#f800fd",  # 6.00 - 8.00 inches
                "#9854c6",  # 8.00 - 10.00 inches
                "#fdfdfd"   # 10.00+
            ]
        levels = [0.1, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
              6.0, 8.0, 10.]
    elif var_name == 'FLUT':
        colors = [
            '#ffffff',  # 250+ W/m^2
            '#55fbfd',  # 245 - 250 W/m^2
            '#46d0e2',  # 240 - 245 W/m^2
            '#4dc1d7',  # 235 - 240 W/m^2
            '#40b3d1',  # 230 - 235 W/m^2
            '#3ca6ca',  # 225 - 230 W/m^2
            '#3a97c0',  # 220 - 225 W/m^2
            '#3179ab',  # 215 - 220 W/m^2
            '#396fad',  # 210 - 215 W/m^2
            '#2e5a9b',  # 205 - 210 W/m^2
            '#2a4f94',  # 200 - 205 W/m^2
            '#293683',  # 195 - 200 W/m^2
            '#221f76',  # 190 - 195 W/m^2
            '#222f6f',  # 185 - 190 W/m^2
            '#254562',  # 180 - 185 W/m^2
            '#245657',  # 175 - 180 W/m^2
            '#2c7150',  # 170 - 175 W/m^2
            '#338c49',  # 165 - 170 W/m^2
            '#3eb53d',  # 160 - 165 W/m^2
            '#44f939',  # 155 - 160 W/m^2
            '#6cf83a',  # 150 - 155 W/m^2
            '#92f840',  # 145 - 150 W/m^2
            '#d8fb3d',  # 140 - 145 W/m^2
            '#fcda40',  # 135 - 140 W/m^2
            '#fb822e',  # 130 - 135 W/m^2
            '#f74127',  # 125 - 130 W/m^2
            '#f6242a',  # 120 - 125 W/m^2
            '#a41622',  # 115 - 120 W/m^2
            '#5c141b',  # 110 - 115 W/m^2
            '#221615',  # 105 - 110 W/m^2
            '#3f3f3f',  # 100 - 105 W/m^2
            '#727175',  #  95 - 100 W/m^2
            '#babbbd',  #  90 -  95 W/m^2
            '#ffffff',  #  85 -  90 W/m^2
        ]
        levels = np.linspace(355, 75, 35)
    # elif colorbar_name == 'goes_16':
    #     colors = [
            
    elif var_name == 'U10':
        cmap = 'jet'
        return cmap
    else:
        raise ValueError('fix me.')
        
    cmap = mcolors.ListedColormap(colors, var_name)    
    norm = mcolors.BoundaryNorm(levels, len(levels))
    
    return cmap

def basin_bboxes(basin_name):
    if basin_name == 'north atlantic':
        west_coord = -100.0+360
        east_coord = -5.0+360
        north_coord = 50.0
        south_coord = 0.0
    elif basin_name == 'south atlantic':
        west_coord = -105.0+360
        east_coord = -5.0+360
        north_coord = 65.0
        south_coord = 0.0
    elif basin_name == 'east pacific':
        west_coord = -180.0+360
        east_coord = -80.0+360
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
        east_coord = -120.0+360
        north_coord = 0.0
        south_coord = -65.0        
    elif basin_name == 'conus':
        west_coord = -130.0+360
        east_coord = -65.0+360
        north_coord = 50.0
        south_coord = 20.0        
    elif basin_name == 'east conus':
        west_coord = -105.0+360
        east_coord = -60.0+360
        north_coord = 48.0
        south_coord = 20.0
    
    lons = (west_coord, east_coord)
    lats = (south_coord, north_coord)
    
    return lons, lats
