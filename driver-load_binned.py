# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:48:54 2018

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


#Initialize:
ALLdata = LD()

#Load the raw data
#destdir = './RawData/RawData_test/'
#LoadType = 'raw'
#ALLdata.LoadData(LoadType,destdir)

#Load Binned Data
destdir = './BinnedData/5min_avg/'
LoadType = 'binned'
ALLdata.LoadData(LoadType,destdir)
#generate the distances adjancecy matrix
dist = gam.GEN_ADJ_MAT(ALLdata)


dirNormType = 'minmax'
ALLdata.NormalizeVals('temp',dirNormType)
ALLdata.NormalizeVals('speed',dirNormType)
ALLdata.NormalizeVals('solar',dirNormType)


#import makeDirSpeedPlots as mDSP
#import makeDiffPlots as mDP
#mDSP.MakeDirSpeedPlots(ALLdata,dist)
#mDP.MakeDiffPlots(ALLdata,dist)