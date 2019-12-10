# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 23:20:02 2019

@author: jmatt
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('interp_points_with_stations.xlsx')

lines = set(df['line'])
"""
for line in lines:
    plt.figure()
    mask = df['line']==line
    data = df.loc[mask,:]
    mask = df['type']=='point'
    points = data.loc[mask,:]
    stations = data.loc[~mask,:]
    plt.plot(points['easting'],points['northing'],'g.-')
    plt.plot(stations['easting'],stations['northing'],'c*-',markersize = 14)
    plt.title(line)
    plt.xlabel('easting')
    plt.ylabel('northing')
    """

line = "BC"
east_var = 'easting_2'
north_var = 'northing_2'
plt.figure()
mask = df['line']==line
data = df.loc[mask,:]
mask = df['type']=='point'
points = data.loc[mask,:]
stations = data.loc[~mask,:]
plt.plot(points[east_var],points[north_var],'g.')
plt.plot(stations[east_var],stations[north_var],'c*',markersize = 14)
plt.plot(data[east_var],data[north_var],'k-')

ann_data = list(zip(data['point'],list(zip(data[east_var],data[north_var]))))
for text,xy in ann_data:
    plt.annotate(text,xy,fontsize=8)
plt.title(line)
plt.xlabel(east_var)
plt.ylabel(north_var)
