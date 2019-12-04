# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:57:04 2019

@author: Andrea
"""

from pyproj import proj, transform
import numpy as np

def lonlat_xy(input_lon, input_lat, epsg_1, epsg_2):
    ''' This function takes:
            input_lon: a numpy array of longitudes 
            input_lat: a numpy array of latitudes
            epsg_1: the EPSG code (as a numeric) of the input coordinates
            epsg_2: the EPSG code (as a numeric) of the output coordinates
        And returns the normalized x and y coordinates in epsg_2
    '''
    # setup projections
    str_1 = 'epsg:' + str(epsg_1)
    str_2 = 'epsg:' + str(epsg_2)
    crs_1 = proj.Proj(init = str_1)
    crs_2 = proj.Proj(init = str_2) 
    
    # cast geographic coordinate pair to the projected system
    x, y = transform(crs_1, crs_2, input_lon, input_lat)
    
    # use built-in array methods to normalise values:
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    print('x: ' + str(x))
    print('y: ' + str(y))
    print('x_norm: ' + str(x_norm))
    print('y_norm: ' + str(y_norm))
    
    return x,y
    
# Run Test Example:
input_lon = np.array([-114.74, -234.23, -116.43, -234.2])
input_lat = np.array([44.06, 43.21, 45.23, 43.33])

input_lon = np.array([-114.74, -116.43])
input_lat = np.array([44.06, 45.23])

epsg_1 = 2241 # NAD83 Idaho East
epsg_2 = 32611 # UTM Zone 11N
x,y = lonlat_xy(input_lon, input_lat, epsg_1, epsg_2)
# Confirmation: https://epsg.io/transform#s_srs=2241&t_srs=32611&x=-114.7400000&y=44.0600000