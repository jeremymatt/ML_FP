#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:41:51 2018

@author: jmatt
"""

import numpy as np
import pandas as pd
import pygrib
import os

#Try to load the actual data extents that cover the INL weather stations; 
#if not available then load sample extents instead
try:
    import HRRR_extents as ext
    lat_min=ext.lat_min
    lat_max=ext.lat_max
    lon_min=ext.lon_min
    lon_max=ext.lon_max
except:
    #Sample lat/lon extents
    lat_min=30
    lat_max=30.25
    lon_min=-100
    lon_max=-99.75

#file where the data is stored
data_path = 'RawData/HRRR/'

#Generate a list of filenames
files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,f))]

#Open the first file
filename = files[0]
grbs = pygrib.open(data_path+filename)

#read the first file to a pygrib object
grb = grbs.read(1)[0]

#Extract the grid of lat/lon values & print description of the extents
all_lats, all_lons = grb.latlons()
print('The data has extents of:')
print('latitude:{} to {}  \nlongitude:{} to {}'.format(all_lats.min(),all_lats.max(),all_lons.min(),all_lons.max()))

#Check that the requested extents are within the available data
test1 = lat_min<all_lats.min()
test2 = lat_max>all_lats.max()
test3 = lon_min<all_lons.min()
test4 = lon_max>all_lons.max()
#Warn user if extents are out-of-bounds
if test1 or test2 or test3 or test4:
    print('WARNING: extents of requested data is outside the bounds of available data:')
    print('Latitude:')
    print('    Requested:{: <20} to {: <20}'.format(lat_min,lat_max))
    print('    Available:{: <20} to {: <20}'.format(all_lats.min(),all_lats.max()))
    print('Longitude:')
    print('    Requested:{: <20} to {: <20}'.format(lon_min,lon_max))
    print('    Available:{: <20} to {: <20}'.format(all_lons.min(),all_lons.max()))

#########
#Check if all the lats/lons are the same in each grid
#########
# If checkLL is false, skip the grid check
checkLL = True
#Init list to store names of inconsistent files
bad_files = []
if checkLL:
    for ind,filename in enumerate(files):
        #Every tenth file, let user know where the process is
        if np.mod(ind,10)==0:
            print('checking file {} of {}'.format(ind,len(files)-1))
        #Open and read the .grib2 file
        grbs = pygrib.open(data_path+filename)
        grb = grbs.read(1)[0]
        #Extract the lat lon data w/in the requested data extents
        data, lats, lons = grb.data(lat1=lat_min,lat2=lat_max,lon1=lon_min,lon2=lon_max)
        #zip the lats and lons into a list of tuples (each tuple is considered to be the name of a location)
        loc_name_curr = list(zip(lats,lons))
        #The first time through the loop, store the name tuples
        if ind==0:
            loc_name=loc_name_curr
        #For each subsequent loop, check if the current name tuples are the same
        #as the names from the first file.  Warn user if different
        elif not loc_name==loc_name_curr:
            print('File {} has different lat/lon values'.format(filename))
            bad_files.append(filename)
        

for ind,filename in enumerate(files):
    #Every tenth file, let user know where the process is
    if np.mod(ind,10)==0:
        print('Loading file {} of {}'.format(ind,len(files)-1))
    #Open and read the .grib2 file
    grbs = pygrib.open(data_path+filename)
    grb = grbs.read(1)[0]
    #Extract the variable name and the datetime of the grib file
    parameter = grb.parameterName
    datetime = grb.validDate
    #Extract the lat lon data w/in the requested data extents
    data, lats, lons = grb.data(lat1=lat_min,lat2=lat_max,lon1=lon_min,lon2=lon_max)
    #zip the lats and lons into a list of tuples (each tuple is considered to be the name of a location)
    loc_name = list(zip(lats,lons))
    #Convert name tuples to strings
    loc_name = ['HRRR_{}_{}'.format(tpl[0],tpl[1]) for tpl in loc_name]
    #Store the extracted values in a dataframe
    temp_df = pd.DataFrame({
            'Data':data,
            'loc_name':loc_name,
            'Param':parameter,
            'datetime':datetime
            })
    
    #Try to append to the dataframe - if that doesn't work init the dataframe
    #with values from the first .grib file
    try:
        data_list = data_list.append(temp_df,ignore_index=True)
    except:
        data_list = temp_df

#Pivot the data into table format with location and parameter as multilevel
#header keys and with the index as the datetime column
data_table = data_list.pivot_table(index='datetime',columns=['loc_name','Param'])


#Directory to save the HRRR data in
s_dir = './DataBinned/HRRR_out/'
#loop through each file
for ind,name in enumerate(loc_name):
    #Tell User where script is in the process
    if np.mod(ind,100)==0:
        print('Saving file {} of {}'.format(ind,len(loc_name)-1))
    #Extract the data associated from the current point in the .grib grid
    cur_pt_data = data_table['Data'][name]
    #define variables for easier indexing
    Vname = 'v-component of wind'
    Uname = 'u-component of wind'
    #Extract the u and v components of windspeed, calc the mag. of the vector
    #and convert from m/s to mph & store in pandas series
    total_windspeed = ((cur_pt_data[Vname]**2+cur_pt_data[Uname]**2)**0.5)*2.23694
    #convert to from series to dataframe
    total_windspeed = pd.DataFrame({'HRRR_ws':total_windspeed})
    #rename the index to be consistent with naming convention for other files
    total_windspeed.index.names = ['datetime_bins']
    #Reset the index to integers (drop defaults to false, so the datetime i
    #index is converted to a data column)
    total_windspeed.reset_index(inplace=True)
    
    #Generate the filename to save the data
    filename = s_dir+'_name-'+name+'.binned'
    #Open the file and write the header info
    file = open(filename,'w')
    file.write('site name:'+name+chr(10))
    file.write('units: mph'+chr(10)+chr(10))
    #flush() required prior to to_csv() for pandas version 0.23.1
    file.flush()
    #Write the data block
    total_windspeed.to_csv(file,mode='a',header=True,index=True,index_label='index')
    #Close the file
    file.close()

#For each name (from the last .grib file), make a list of tuples containing:
#   1. The name string
#   2. The latitude
#   3. The longitude
names_locs = [(name,name.split('_')[1],name.split('_')[2]) for name in loc_name]
#Generate a dataframe
names_locs = pd.DataFrame(names_locs)
#Name the columns
names_locs.rename(columns={0:'Name',1:'lat',2:'lon'},inplace=True)
#Export the name/lat/lon data to .csv
file = open('HRRR_locs.csv','w')
names_locs.to_csv(file,mode='a',header=True,index=True,index_label='index')
file.close()
