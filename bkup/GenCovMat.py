# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:50:24 2018

@author: MATTJE
"""

import numpy as np
import pandas as pd
import copy as cp

def GenCovMat(ALLdata):
    #List of the names of the variables
    names = ['temp:','solar:','speed:','dir:']
    #new names for the variables of the first station
    newnames = ['S0_'+x for x in names]
    #Extract the data for the first station
    CompiledData = cp.copy(ALLdata.WSdata[0].data_binned)
    #Set the datetime field to be the index
    CompiledData.set_index('datetime_bins',inplace=True)
    #rename the colums
    CompiledData.columns = newnames
    #For each of the remaining stations, grab the data and add to the dataframe
    for i in range(1,ALLdata.numFiles):
        #Get the data for Station i
        addset = cp.copy(ALLdata.WSdata[i].data_binned)
        #Set the datetime field to be the index
        addset.set_index('datetime_bins',inplace=True)
        #Generate column names for Station i
        newnames = ['S'+str(i)+'_'+x for x in names]
        #Rename the columns
        addset.columns = newnames
        #Join the data
        CompiledData = CompiledData.join(addset,how='outer')
    #Return the covariance matrix
    return CompiledData.cov()