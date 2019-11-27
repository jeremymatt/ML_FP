
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

#Import user-defined functions
from geostats_functions import *



#Initialize the data object:
ALLdata = LD()

#Location of the data to be loaded
destdir = './DataBinned/HRRR_INL-5minraw_noS35_no-HRRR/'
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
start = '2017-01-01 12:15:00'
end = '2017-01-01 12:15:00'

#Specify the list of stations to load
#ALLdata.WSdata is a list of objects that hold the data for each station
#Another valid call would be something like ALLdata.WSdata[:10], which would
#load the data for the first 10 stations
stations = ALLdata.WSdata 

#Generate a dataframe holding the data between the start and end times
data = ALLdata.get_krig_data(stations,start,end)

