# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:16:16 2019

@author: jmatt
"""


import scipy.stats as sps
from statsmodels.graphics.gofplots import qqplot
from select_daterange import select_daterange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def station_QQ_plots(stations,start,end,var):
    """
    INPUTS
        stations - list of station objects
        start - tuple or list in the form of (year,month,day)
        end - tuple or list in the form of (year,month,day)
        var - the variable to make the QQ plots for
    """
    
    num_stations = len(stations)
    cols = int(np.round(num_stations**.5))
    rows = int(np.ceil(num_stations/cols))
    
    
    fig, axes = plt.subplots(rows,cols,sharex = True,sharey = True,figsize=(13, 13), dpi= 80, facecolor='w', edgecolor='k')
    zipped = list(zip(stations,axes.ravel()[:num_stations]))
    mode = 'hard'
    
    for station,ax in zipped:
        data = pd.DataFrame(station.data_binned[['datetime_bins',var]])
        data.set_index('datetime_bins',inplace=True)
        
        selected = select_daterange(data,start,end,mode)
        qqplot(selected[var],line='s',ax=ax)  
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.title.set_text(station.name)
        k2,p = sps.kstest(selected[var],'norm')
        ax.annotate('K-S p:{:0.3g}'.format(p),xy = (0.05,.9),xytext = (0.05,.9), textcoords='axes fraction',horizontalalignment='left', verticalalignment='top')
        fig.suptitle(var.split(':')[0])
     
    
    