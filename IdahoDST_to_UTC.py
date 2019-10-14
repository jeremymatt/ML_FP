#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:42:36 2018

@author: jmatt
"""

import pandas as pd


def IdahoDST_to_UTC(df,MST_offset):
    """
    Simple conversion from Idaho DST to UTC by simply adding/subtracting the 
    appropriate number of hours.  
    
    *****NOTE*******
    This fcn does NOT address ambiguous times (IE 1:30AM) - instead it simply
    shifts all values that are w/in the dst range back 1 hour.
    This works for the weather station data I have because there are no 
    duplicated times - my best guess is that after the switch back from DST 
    the system just over-wrote an hours' worth of data
    
    INPUTS:
        df - A dataframe with a datetime index that needs to be converted
        
    """
    #make a copy of the dataframe
    df = pd.DataFrame(df)
    
#    MST_offset = -7  #UTC-7
    #Define points at which system switches to/from DST
    DST_switch = [
            (pd.to_datetime('2015-03-08 02:00'),pd.to_datetime('2015-11-01 01:59')),
            (pd.to_datetime('2016-03-13 02:00'),pd.to_datetime('2016-11-06 01:59')),
            (pd.to_datetime('2017-03-12 02:00'),pd.to_datetime('2017-11-05 01:59')),
            (pd.to_datetime('2018-03-11 02:00'),pd.to_datetime('2018-11-04 01:59')),
            (pd.to_datetime('2019-03-10 02:00'),pd.to_datetime('2019-11-03 01:59'))]
    
    #rename the index to datetime & reset the index to make the datetime a column
    df.index.name = 'datetime'
    df = df.reset_index()
    breakhere=1
    #Find the min and max dates in the list of DST switches
    min_covered = min(min(DST_switch))
    max_covered = max(max(DST_switch))
    #Find the min/max dates of the data
    min_data_date = df['datetime'].min()
    max_data_date = df['datetime'].max()
    #If the data are outside the range of the list of DST switches, warn user
    if (min_data_date<min_covered):
        print('WARNING: The earliest date in the data <{}> is before the min date <{}> that is converted'.format(min_data_date,min_covered))
        print('Consider adding entries to the DST_switch variable')
    if (max_data_date>max_covered):
        print('WARNING: The latest date in the data <{}> is after the max date <{}> that is converted'.format(max_data_date,max_covered))
        print('Consider adding entries to the DST_switch variable')
    
    #Grab the first DST switch
    dst = DST_switch[0]
    #Find places where the data matches the DST switch
    mask_final = (df['datetime']>=dst[0]) & (df['datetime']<=dst[1])
    
    #for each subsequent switch, check for additional values and update the 
    #mask
    for dst in DST_switch[1:]:
        mask_temp = (df['datetime']>=dst[0]) & (df['datetime']<=dst[1])
        mask_final = mask_final | mask_temp
        
    #For data that are in DST, subtract 1 hour
    df.loc[mask_final,'datetime'] = df.loc[mask_final,'datetime'] - pd.DateOffset(hours=1)
    #For all data, add 7 hours
    df.loc[:,'datetime'] = df.loc[:,'datetime'] + pd.DateOffset(hours=MST_offset)
    #reset the index
    df.set_index('datetime',inplace=True)
    #return the dataframe
    return df
