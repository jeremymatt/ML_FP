
# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19

import matplotlib as mpl
import pandas as pd
import time
from weather_loaddata import WEATHER_LOADDATA as LD
import geostats_functions as GSF
import predict_time_remaining as PTR
from sklearn.preprocessing import StandardScaler


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
ALLdata.calc_wind_u_v(scale_by_speed=False)

#Load the northing/easting data in to the data objects for each station
filename = 'northing_easting.xlsx'
ALLdata.load_xyz(filename)

#%%
#Specify the start and end times
start = '2017-01-01 11:00:00'
end = '2017-02-01 11:15:00'


start = pd.to_datetime(start)
end= pd.to_datetime(end)

#Specify the times within each day to grab times from.  
#NOTE: 
start_hour = 13
end_hour = 13
sub_hour_range = True
start_minute = 0
end_minute = 0

if sub_hour_range:
    filename = 'jan2017_{}_pairs'.format(start_hour)
else:
    filename = 'jan2017_{}-{}_pairs'.format(start_hour,end_hour)


data = ALLdata.WSdata[0].data_binned.copy()
data['hr'] = data['datetime_bins'].dt.hour
data['min'] = data['datetime_bins'].dt.minute

m1 = data['datetime_bins']>=start
m2 = data['datetime_bins']<end
m3 = data['hr']>=start_hour
m4 = data['hr']<=end_hour
if sub_hour_range:
    m5 = data['min']>=start_minute
    m6 = data['min']<=end_minute
else:
    m5 = data['min']==data['min']
    m6 = m6



times_to_process = list(data.loc[m1&m2&m3&m4&m5&m6,'datetime_bins'].values)
times_to_process = data.loc[m1&m2&m3&m4&m5&m6,'datetime_bins']
num_times = len(times_to_process)


#Specify the list of stations to load
#ALLdata.WSdata is a list of objects that hold the data for each station
#Another valid call would be something like ALLdata.WSdata[:10], which would
#load the data for the first 10 stations
stations = ALLdata.WSdata 

x = 'easting'
y = 'northing'
variables = ['solar:','temp:','speed:','dir:','u','v']

var_data = ALLdata.WSdata[0].data_binned[variables].values

scaler = StandardScaler()
scaler.fit(var_data)

scaler_vars = (scaler,variables)
scaler_vars = None

generate = True
if generate:
    paired_data = pd.DataFrame()
    
    for ind,cur_time in enumerate(times_to_process):
        tic = time.time()
        #Generate a dataframe holding the data between the start and end times
        data = ALLdata.get_krig_data(stations,cur_time,cur_time,scaler_vars = scaler_vars)
        
        pairs = GSF.extract_pairs(data, x, y, variables, unordered=True)
        
        paired_data = paired_data.append(pairs)
        
        toc = time.time()
        
        
        PTR.predict(tic,toc,ind,num_times)
     
    half_length = int(paired_data.shape[0]/2)
    paired_data.reset_index(inplace=True,drop=True)
    print('\n\nSaving Files')
    #Max rows in an excel sheet
    #split into two files
    if paired_data.shape[0]>1048575: 
        tic = time.time()
        paired_data.iloc[:half_length,:].to_excel('{}_part1.xlsx'.format(filename))
        toc = time.time()
        PTR.predict(tic,toc,1,2)
        paired_data.iloc[half_length:,:].to_excel('{}_part2.xlsx'.format(filename))
    else:
        paired_data.to_excel('{}.xlsx'.format(filename))
