# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:15:32 2018

@author: MATTJE
"""
#Script to export plots of all values over time.

#####
#####  NOTE: Requires that the ALLdata object be initialized - run driver.py first
#####


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

names = {}
names[0] = 'temp'
names[1] = 'solar'
names[2] = 'speed'
names[3] = 'dir'


for i in range(0,ALLdata.numFiles):
    data = ALLdata.WSdata[i].data_binned
    for ii in range(0,4):
        title = names[ii] + ' at '+ '(Station ID-'+str(i)+')' + ALLdata.WSdata[i].name
        
        y = np.array(data[names[ii]+':'])
        dr = pd.DatetimeIndex(data['datetime_bins'])
        df = pd.DataFrame({names[ii]:y},index=dr)
        
        plt.plot(data['datetime_bins'],data[names[ii]+':'])
        
        plt.title(title)
        plt.savefig(title+'.png')
        plt.cla()
        
#for i in range(0,ALLdata.numFiles):
#    
#    data = ALLdata.WSdata[i].data
#    title = 'speed_low at '+ '(Station ID-'+str(i+1)+')' + ALLdata.WSdata[i].name 
#    plt.scatter(data['rec nr:'],data['speed:'])
#    
#    
#    y = np.array(data['speed:'])
#    dr = pd.DatetimeIndex(data['datetime_bins'])
#    df = pd.DataFrame({names[ii]:y},index=dr)
#        
#    
#    plt.ylim([0,5])
#    plt.title(title)
#    plt.savefig(title+'.png')
#    plt.cla()