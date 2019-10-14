# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:56:54 2018

@author: jmatt
"""
import pandas as pd
from make_semivariogram import make_semivariogram
from weather_loaddata import WEATHER_LOADDATA as LD
import gen_adj_mat as gam
import matplotlib.pyplot as plt
import numpy as np

try:
    ALLdata
    dist
except:
    #Initialize:
    ALLdata = LD()
    #Load the binned data
    destdir = './DataBinned/HRRR_INL-5minraw_noS35_no-HRRR/'
    LoadType = 'binned'
    ALLdata.LoadData(LoadType,destdir)
    
#    MST_offset = -7
    MST_offset = 0
    ALLdata.ConvertTimeToUTC(MST_offset)
    
    UTC_dups = [station.NumDupsDropped_UTC for station in ALLdata.WSdata if station.name.split('n')[0] == 'Statio']
    plt.hist(UTC_dups)


    #generate the distances adjancecy matrix
    try: dist = pd.read_csv('distances.csv').set_index('ID')
    except: 
        FileName = 'stationcoords.csv'
        dist = gam.GEN_ADJ_MAT(ALLdata,FileName)
        file = open('distances.csv','w')
        dist.to_csv(file,mode='a',header=True,index=True,index_label='ID')
        file.close()


mode = 'hard'
station_groups = [
        [0,1,2,3,4],
        [5,6,7,8,9],
        [10,11,12,13,14],
        [15,16,17,18,19],
        [20,21,22,23,24],
        [25,26,27,28,29],
        [30,31,32,33,34],
        [36,37,38,39,40],
        [42,43,44,45]]
Solar_groups = [
        [0,1,2,3,4],
        [5,6,7,8,9],
        [10,11,12,13,14],
        [15,16,17,18,19],
        [20,21,22,23,24],
        [25,26,27,28,29],
        [30,31,32,33,34],
        [36,37,38,39,40],
        [42,43,44,45]]

#Direction sets
Dir_groups = [
        [0,1,16,17,19,24,30,36],
        [2,3,4,5,25,38,40,44,45],
        [8,9,10,11,12,13,14,21,31,34],
        [6,7,18,20,26,27,42,43],
        [15,22,23],
        [28,29,32,33],
        [37,39]]

Speed_groups = [
        [0,1,2,4,7,8,11,39,44],
        [3,32],
        [6,9,10],
        [5],
        [15,19],
        [12,13,14,33,34,43],
        [23,24],
        [21,22,36],
        [16,20,25,26,27,28,29,31,38,40,42,45],
        [17,18,30],
        [37]]

Temp_groups = [
        [0,1,16,21],
        [2,4,5,6,8,9,27,28,29,33,37,42,44,45],
        [3,32,38,39,40],
        [7,10,11,12,13,14,15,17,18,19,20,22,23,24,25,26,30,31,34,36,43]]

all_stations = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40,42,43,44,45]]

#Define the list of variables to generate the semivariograms for
var_list = ['speed:','temp:','solar:','dir:']
var_list = ['dir:']
date = [2017,4,1]
#date2 = [2017,4,2]
date_range = [date,date]

if var_list[0] == 'solar:':
    station_groups = Solar_groups
if var_list[0] == 'temp:':
    station_groups = Temp_groups
if var_list[0] == 'speed:':
    station_groups = Speed_groups
if var_list[0] == 'dir:':
    station_groups = Dir_groups
    
#station_groups=all_stations
#station_groups = [[0]]
save_fn = 'SV_plots\\temporal\\{}-{:02d}-{:02d}'.format(date[0],date[1],date[2])

#all_stations = []
#temp = [[all_stations.append(x) for x in lst] for lst in station_sets]
#
#station_sets = [all_stations]

all_stations = [[x for lst in station_groups for x in lst]]

print('Sum of station numbers should be 959.  Sum is actually: {}'.format(np.sum(all_stations)))
station_sets = [['Station{:03d}'.format(st_num) for st_num in lst] for lst in station_groups]

#station_sets = [[0,1,2,3,4]]

"""
Define the semivariogram bin mode
    'bins' - defines the number of bins to use.  Either a fixed number,
            a user-specified list of bin divisions, or 'sqrt'
            If 'sqrt', sets the number of bins equal to the square root of the
            number of differences
    'mode' - defines the binning mode.  Options:
            'eq.pts' - equal number of points in each bin
            'eq.dist' - each bin is of equal width
            'user.div' - user-defined bin divisions
            
"""
sv_mode = {'bins':'sqrt','mode':'eq.pts'}

#Define the list of variables to generate the semivariograms for

#If using a large number of model points as inputs, can utilize only a fraction
fraction_to_keep = 1/20
#Loop through each variable and generate the semivariograms
sv_type = 'time'
for stations_list in all_stations:
    stations=[ALLdata.StationNamesIDX['Station{:03d}'.format(st_num)] for st_num in stations_list]
    
    
    
    for ind,var1 in enumerate(var_list):
        variables = [var1,var1]
        data1,station_list,differences = make_semivariogram(save_fn,ALLdata,sv_type,dist, fraction_to_keep, variables,date_range,mode,station_sets,stations,sv_mode)
#
#        
