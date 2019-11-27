
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



#Initialize:
ALLdata = LD()

#Load the raw data
destdir = './RawData/all/'
LoadType = 'raw'
#Load the binned data
destdir = './DataBinned/HRRR_INL-5minraw_noS35_no-HRRR/'
LoadType = 'binned'


ALLdata.LoadData(LoadType,destdir)
ALLdata.ConvertTimeStrToDatetime()

ALLdata.calc_wind_u_v(scale_by_speed=True)

filename = 'northing_easting.xlsx'
ALLdata.load_xyz(filename)

start = '2017-01-01 12:15:00'
end = '2017-01-01 12:15:00'
data = ALLdata.get_krig_data(ALLdata.WSdata,start,end)

