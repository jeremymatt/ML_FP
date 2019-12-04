# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:20:06 2019

@author: jmatt
"""

from LatLong_xy import *

import pandas as pd

data1 = pd.read_excel('station_locns.xlsx')
data2 = pd.read_excel('midpoint_locns.xlsx')

lat1 = data1['Lat'].values
lon1 = data1['Long'].values
ID1 = data1['new_name'].values

lat2 = data2['Lat'].values
lon2 = data2['Long'].values
ID2 = data2['new_name'].values


epsg_1 = 2241 # NAD83 Idaho East

epsg_1 = 26969 #NAD83 idaho central
epsg_1 = 2888 #NAD83 Idaho West
epsg_2 = 32611 # UTM Zone 11N

epsg_2 = 4326 #lat/lon

northing = []
easting = []

x1,y1 = lonlat_xy(lon1,lat1,epsg_2,epsg_1)
x2,y2 = lonlat_xy(lon2,lat2,epsg_2,epsg_1)

#convert to km
x1 = x1*0.0003048
y1 = y1*0.0003048

x2 = x2*0.0003048
y2 = y2*0.0003048

x_offset = min(x1.min(),x2.min())
y_offset = min(y1.min(),y2.min())


northing_easting = pd.DataFrame({'point':ID1,
                                 'northing':y1-y_offset,
                                 'easting':x1-x_offset,
                                 'elevation':data1['Elevation(feet)'].values})
northing_easting.to_excel('northing_easting.xlsx')

plt.plot(x2-x_offset,y2-y_offset,'g.',label = 'Span Center')
mask = [True if val.split('n')[0]=='Statio' else False for val in ne['point'].values]
plt.plot(northing_easting.loc[mask,'easting'],northing_easting.loc[mask,'northing'],'b*',markersize=12,label = 'Weather Station')
plt.xlabel('easting(km)')
plt.ylabel('northing(km)')
plt.legend()
plt.savefig('stations.png')


northing_easting = pd.DataFrame({'point':ID2,
                                 'northing':y2-y_offset,
                                 'easting':x2-x_offset,
                                 'elevation':data2['Elevation(feet)'].values})
northing_easting.to_excel('interp_points.xlsx')
