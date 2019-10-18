# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:05:33 2018

@author: jmatt
"""
import pandas as pd
import copy as cp
def select_daterange(input_dataframe,Start,End,mode):
    """
    Select data from a pandas dataframe between start and end date.
    Sample calls:
        select_daterange(df,[2018,5,1],[2019,1,15],'hard')
            Select all data between 2018-5-1 and 2019-1-15
        select_daterange(df,[2018,5,1],[2019,5,5],'seasonal')
            Select all data with a date of 5/1,5/2,5/3,5/4, and 5/5 irrespective
            of year
            
    input_dataframe - pandas dataframe with datetimestamp index
    Start/End - tuple/list of three integers: [yr,mo,day]
    mode -  'hard' (all data between start and end)
            'seasonal' (all data between start and end ignoring year)
    """

    #Convert start and end to strings if they are not already in that format
    if len(Start)>1:
        Start = str(Start[0])+'-'+str(Start[1])+'-'+str(Start[2])
        
    if len(End)>1:
            End = str(End[0])+'-'+str(End[1])+'-'+str(End[2])
    
    #make sure the input dataframe has datetime index
    input_dataframe.index = pd.to_datetime(input_dataframe.index)
    
    if mode == 'hard':
        #Select data and put into variable to return
        breakhere=1
        data = cp.copy(input_dataframe[Start:End])  
    elif mode == 'seasonal':
        #Convert start/end to datetime objects
        Start = pd.to_datetime(Start)
        End = pd.to_datetime(End)
        #Record the keys of the input dataframe
        df_keys = input_dataframe.keys()
        #Make a temp datetime column from the index
        input_dataframe['dt'] = input_dataframe.index
        #Make temp month/day column from the datetime column
        input_dataframe['month'] = input_dataframe['dt'].dt.month
        input_dataframe['day'] = input_dataframe['dt'].dt.day
        #Mask off locations where the month/day are after the month/day of Start
        m1 = (input_dataframe['month']>Start.month) | ((input_dataframe['month']==Start.month) & (input_dataframe['day']>=Start.day))
        #Mask off locations where the month/day are before the month/day of End
        m2 = (input_dataframe['month']<End.month) | ((input_dataframe['month']==End.month) & (input_dataframe['day']<=End.day))
        #Compile the finished mask
        mask = m1 & m2
        #Select the data of interest
        data = cp.copy(input_dataframe[mask]) 
        #Remove the temp datetime, month, and day columns
        data = data[df_keys]
        
    else:
        #Check for invalid mode
        print('ERROR: invalid select mode')
        return
      
    #Return the data
    return data

