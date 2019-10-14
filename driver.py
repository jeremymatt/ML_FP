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


#MST_offset = 7
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
    
    
StartDate = '12/19/2016'
EndDate = '12/22/2016'
#StartDate = '3/30/2017'
#EndDate = '4/2/2017'
Station = {}
Station[0] = 15
Station[1] = 22 
Station[2] = 23
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
    
    
    
    
    
    
    
    
    

if LoadType == 'raw':
    ALLdata.CheckReadingRanges()
    #Remove all records where all sensor values are zero
    ALLdata.RemoveAllZeros()
    #Check the time spacing between each reading in the data set
    ALLdata.CheckTimeSpace()
    
    #Determine the minimum stepsize for which at least CutoffPercent of the 
    #readings have an equal or smaller stepsize
    CutoffPercent = 99
    mask = ALLdata.TimeStepSum['% < or ==']>CutoffPercent
    MinStepSize = np.array(ALLdata.TimeStepSum['StepSize(min)'][mask])[0]
    
    
    timestep = 5 #minutes
    options = 'avg'
    firstYear = 2016 #the first year in the dataset
    ALLdata.BinDataSets(timestep,options,firstYear)
    print(ALLdata.BinTimestepSum)
    ALLdata.SaveBinnedData()
    
#    dirNormType = 'minmax'
#    ALLdata.NormalizeVals('temp',dirNormType)
#    ALLdata.NormalizeVals('speed',dirNormType)
#    ALLdata.NormalizeVals('solar',dirNormType)
    
    
#    import makeDirSpeedPlots as mDSP
#    import makeDiffPlots as mDP
#    mDSP.MakeDirSpeedPlots(ALLdata,dist)
#    mDP.MakeDiffPlots(ALLdata,dist)
    
    
#    runfile('C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2/driver-PlotDiurnals.py', wdir='C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2')
    
