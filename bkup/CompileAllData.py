# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:50:24 2018

@author: MATTJE
"""

import numpy as np
import pandas as pd
import copy as cp

def CompileAllData(ALLdata,JoinMethod='outer'):
    #outer - keep all of the data and insert NaN for values that don't exist
    #inner - keep only records where data exists for all stations
    names = ['temp:','solar:','speed:','dir:']
    
    newnames = ['S0_'+x for x in names]
    
    CompiledData = cp.copy(ALLdata.WSdata[0].data_binned)
    CompiledData.set_index('datetime_bins',inplace=True)
    CompiledData.columns = newnames
    
    for i in range(1,ALLdata.numFiles):
        addset = cp.copy(ALLdata.WSdata[i].data_binned)
        addset.set_index('datetime_bins',inplace=True)
        newnames = ['S'+str(i)+'_'+x for x in names]
        addset.columns = newnames
        CompiledData = CompiledData.join(addset,how=JoinMethod)
        breakhere=1
        
#    output = CompiledData.cov()
        
    return CompiledData