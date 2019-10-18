# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:27:01 2019

@author: jmatt
"""

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

def spatial_QQ_plots(ALLdata,timestamps):
    """
    INPUTS
        stations - list of station objects
        start - tuple or list in the form of (year,month,day)
        end - tuple or list in the form of (year,month,day)
        var - the variable to make the QQ plots for
    """
    
    labels = ['temp','direction','speed','solar']
    
    num_times = len(timestamps)
    
    
    fig, axes = plt.subplots(num_times,4,figsize=(13, int(np.round(num_times*13/4))), dpi= 80, facecolor='w', edgecolor='k')
    
    
    for row,ts in enumerate(timestamps):
        temp = []
        wind_dir = []
        wind_speed = []
        solar = []
        
        for station in ALLdata.WSdata:
            df = station.data_binned.loc[ts]
            temp.append(df['temp:'])
            wind_dir.append(df['dir:'])
            wind_speed.append(df['speed:'])
            solar.append(df['solar:'])
        
        
        
        lst = [np.matrix(sorted(temp)).T,np.matrix(sorted(wind_dir)).T,np.matrix(sorted(wind_speed)).T,np.matrix(sorted(solar)).T]
        
        for col,vals in enumerate(lst):
            ax = axes[row,col]
            qqplot(vals,line='s',ax=ax) 
            ax.set_xlabel('')
            ax.set_ylabel('')
            k2,p = sps.shapiro(vals)
            p_str ='{:.2f}'.format(p)
            print('timestamp:{}, var:{}, p: {}'.format(ts,labels[col],p_str))
            ax.set_xlabel('Shap p:{}'.format(p_str))
            #ax.annotate('Shap p:{}'.format(p_str),xy = (0.05,.9),xytext = (0.05,.9), textcoords='axes fraction',horizontalalignment='left', verticalalignment='top')
            #ax.annotate('p'.format(p),xy = (0.05,.9),xytext = (0.05,.9), textcoords='axes fraction',horizontalalignment='left', verticalalignment='top')
            if col==0:
                ax.set_ylabel(ts)
            if row==0:
                ax.title.set_text(labels[col])

    fig.tight_layout(pad = 1.1)
      
    