# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:47:11 2019

@author: jmatt
"""

import numpy as np
import pandas as pd

station_list = ALLdata.WSdata[0:1]

station = station_list[0]
station.name
data = station_list[0].data_binned


lower = pd.to_datetime('2017-01-01 00:00:00')
upper = pd.to_datetime('2017-03-01 00:00:00')

start = pd.to_datetime('2017-08-09 11:00:00')
step = pd.to_timedelta('00:20:00')
end = start+step

m1 = data['datetime_bins']>=lower
m2 = data['datetime_bins']<upper
data_block = data[m1&m2]

grab_vars = ['temp:','solar:','u','v']

data_vector = data_block[grab_vars].values.ravel().tolist()
data_vector
t = [lower].extend(data_vector)
t


headers = ['ts{}|{}'.format(step,var) for step in range(data_block.shape[0]) for var in grab_vars]

feat_vect = pd.DataFrame({'time':lower,headers:data_vector})
