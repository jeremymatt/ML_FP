# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:38:44 2018

@author: MATTJE
"""


import numpy as np
import pandas as pd
import os
import time
import sklearn.cluster as cl
import matplotlib.pyplot as plt
from weather_loaddata import WEATHER_LOADDATA as LD
import pickle
import copy as cp
import gen_adj_mat as gam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PlotDataRange as PDR
import scipy.stats as sps
from statsmodels.graphics.gofplots import qqplot
from select_daterange import select_daterange
from station_QQ_plots import station_QQ_plots
from spatial_QQ_plots import spatial_QQ_plots

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
#ALLdata.Check_DST()





#MST_offset = 0
#ALLdata.ConvertTimeToUTC(MST_offset)



#Data is loaded into WSData objects with the following parameters:
ALLdata.WSdata[0].__dict__.keys()
ALLdata.WSdata[0].name
ALLdata.WSdata[0].data_binned.head(5)

#generate the distances adjancecy matrix
try: dist = pd.read_csv('distances_INL-only.csv').set_index('ID')
except: 
    FileName = 'stationcoords.csv'
    dist = gam.GEN_ADJ_MAT(ALLdata,FileName)
    file = open('distances.csv','w')
    dist.to_csv(file,mode='a',header=True,index=True,index_label='ID')
    file.close()
    
    
StartDate = '1/2/2017'
EndDate = '1/3/2017'
#StartDate = '3/30/2017'
#EndDate = '4/2/2017'
Station = {}
Station[0] = 0
Station[1] = 1 
Station[2] = 2
#Station[3] = 19
#Station[4] = 24
#Station[5] = 30 
#Station[6] = 36 
#Station[7] = 21 
#Station[8] = 31 
#Station[9] = 34 
#Station = [32]
ReadingType = {}
ReadingType[0] = 'dir'
#ReadingType[1] = 'speed'
Station = ['Station{:03d}'.format(Station[i]) for i in Station.keys()]
PDR.PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType)
    
run_descriptive_statistics = False
if run_descriptive_statistics:   
    var = 'solar:'
    data = pd.DataFrame(ALLdata.WSdata[15].data_binned[['datetime_bins',var]])
    data.set_index('datetime_bins',inplace=True)
    
    
    start = [2015,3,1]
    end = [2019,4,2]
    var = 'temp:'
    stations = ALLdata.WSdata
    station_QQ_plots(stations,start,end,var)
    
    
    for station in ALLdata.WSdata:
        station.data_binned.set_index('datetime_bins',inplace=True)
        
    print('test')
        
    times = []
    #times.append('2017-04-01 00:00:00')
    #times.append('2017-04-01 08:00:00')
    #times.append('2017-04-01 12:00:00')
    #times.append('2017-04-01 20:00:00')
    
    times.append('2017-01-01 00:00:00')
    times.append('2017-01-01 09:00:00')
    times.append('2017-01-01 12:00:00')
    times.append('2017-01-01 20:00:00')
    
    timestamps = [pd.to_datetime(time) for time in times]
    spatial_QQ_plots(ALLdata,timestamps)
    
    
    num_binned_readings = [station.data_binned.shape[0] for station in ALLdata.WSdata]
    plt.hist(num_binned_readings,bins=8)
    plt.xlabel('# data points per station')
    plt.ylabel('# stations')
               


#     
#
#if LoadType == 'raw':
#    ALLdata.CheckReadingRanges()
#    #Remove all records where all sensor values are zero
#    ALLdata.RemoveAllZeros()
#    #Check the time spacing between each reading in the data set
#    ALLdata.CheckTimeSpace()
#    
#    #Determine the minimum stepsize for which at least CutoffPercent of the 
#    #readings have an equal or smaller stepsize
#    CutoffPercent = 99
#    mask = ALLdata.TimeStepSum['% < or ==']>CutoffPercent
#    MinStepSize = np.array(ALLdata.TimeStepSum['StepSize(min)'][mask])[0]
#    
#    
#    timestep = 5 #minutes
#    options = 'avg'
#    firstYear = 2016 #the first year in the dataset
#    ALLdata.BinDataSets(timestep,options,firstYear)
#    print(ALLdata.BinTimestepSum)
#    ALLdata.SaveBinnedData()
    
#    dirNormType = 'minmax'
#    ALLdata.NormalizeVals('temp',dirNormType)
#    ALLdata.NormalizeVals('speed',dirNormType)
#    ALLdata.NormalizeVals('solar',dirNormType)
    
    
#    import makeDirSpeedPlots as mDSP
#    import makeDiffPlots as mDP
#    mDSP.MakeDirSpeedPlots(ALLdata,dist)
#    mDP.MakeDiffPlots(ALLdata,dist)
    
    
#    runfile('C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2/driver-PlotDiurnals.py', wdir='C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2')
    
