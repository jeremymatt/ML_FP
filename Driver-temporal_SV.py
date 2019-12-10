
# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19

import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from weather_loaddata import WEATHER_LOADDATA as LD
import gen_adj_mat as gam
import PlotDataRange as PDR
import geostats_functions as GSF
import predict_time_remaining as PTR

#Import user-defined functions
from geostats_functions import *



#Initialize the data object:
ALLdata = LD()

#Location of the data to be loaded
destdir = './DataBinned/HRRR_INL-1hr_raw_noS35_no-HRRR/'
#load pre-binned data
LoadType = 'binned'

#Load the data
ALLdata.LoadData(LoadType,destdir)
#Convert the times to datetime objects
ALLdata.ConvertTimeStrToDatetime()

#Add variables for the U and V components of wind
ALLdata.calc_wind_u_v(scale_by_speed=True)

#Load the northing/easting data in to the data objects for each station
filename = 'northing_easting.xlsx'
ALLdata.load_xyz(filename)

#Specify the start and end times
start = '2017-01-01 11:00:00'
end = '2017-02-01 13:00:00'

start = pd.to_datetime(start)
end= pd.to_datetime(end)

station = ALLdata.WSdata[0]

data = station.data_binned

m1 = data['datetime_bins']>start
m2 = data['datetime_bins']<end

data = data.loc[m1&m2,:].copy()
data.reset_index(inplace=True,drop=True)

data['y'] = 0
x = 'datetime_bins'
y = 'y'


#Specify the list of stations to load
#ALLdata.WSdata is a list of objects that hold the data for each station
#Another valid call would be something like ALLdata.WSdata[:10], which would
#load the data for the first 10 stations
stations = ALLdata.WSdata 

variables = ['solar:','temp:','speed:','dir:','u','v']

    
pairs = GSF.extract_pairs(data, x, y, variables)
    
    
