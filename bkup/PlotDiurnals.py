1
in the# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:53:06 2018

@author: MATTJE
"""

import numpy as np
import pandas as pd
import copy as cp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import dates as d
import datetime as dt
import numbers as num


#Plots the average daily values (averaged over a date range for all stations)
#   ALLdata - object containing the input data
#   Ranges - dataframe with columns "StartDate" and "EndDate".  Each row is a
#           range to plot separately
#   Labels - dict containing the labels for the ranges
#   Readings - dict containing the readings (temp, speed, solar, dir) to prepare a plot for
def PlotDiurnals(ALLdata,Ranges,Labels,Readings,Stations,TitleOverride,font_size = 16):
    #Find the number of sub-plots to generate (one for each reading type)
    NumPlots = len(Readings)
    #Find the number of ranges (IE winter/summer) to plot
    NumRanges = len(Labels)
    #Initialize the figure
    f, axarr = plt.subplots(NumPlots, sharex=True, figsize=(13, 11), dpi= 80, facecolor='w', edgecolor='k')
    #If only doing 1 diurnal, convert axarr to a list of size 1
    if NumPlots==1:
        axarr = [axarr]
        
    #Add ylabels to each of the plots
    for i in range(0,NumPlots):
        axarr[i].set_ylabel(Readings[i]+' Profile', fontsize=14, weight='bold')
      
    #Set the xlabel for the bottom diurnal
    axarr[-1].set_xlabel('Time of Day', fontsize=font_size)
    #Set up a colors variable for each diurnal range
    colors = cm.rainbow(np.linspace(0, 1, NumRanges+2))
    #Make the list of headers to grab (each of the variables and the datestamp)
    headers = [Readings[x]+':' for x in Readings]
    #Add the datetime field to the headers list
    headers.append('datetime_bins')
    
    #For each of the date ranges
    for i in range(0,NumRanges):
        #init list to hold station numbers
        StatNum = []
        #Compile all the data from the stations to be aggregated
        if Stations=='all':
            #Start the data compiled structure by grabbing info from station 0
            DataCompiled = cp.copy(ALLdata.WSdata[0].data_binned[headers])
            #For each of the remaining stations, add the data to the bottom of the dataframe
            for ii in range(1,ALLdata.numFiles):
                DataCompiled = DataCompiled.append(ALLdata.WSdata[ii].data_binned[headers],ignore_index=True)
            #make month and day columns
            DataCompiled['Month'] = DataCompiled['datetime_bins'].dt.month
            DataCompiled['Day'] = DataCompiled['datetime_bins'].dt.day
        else:
            #For each of the listed stations
            for x in Stations:
                #If the station is given as a number
                if isinstance(x,num.Number):
                    #add it to the list of station numbers
                    StatNum.append(x)
                else:
                    #else grab the station number from the index
                    StatNum.append(ALLdata.StationNamesIDX[x])
            #Init the data compiled structure with the data from the first station
            DataCompiled = cp.copy(ALLdata.WSdata[StatNum[0]].data_binned[headers])
            #For each of the remaining stations, add the data to the bottom of the dataframe
            for ii in range(1,len(StatNum)):
                DataCompiled = DataCompiled.append(ALLdata.WSdata[StatNum[ii]].data_binned[headers],ignore_index=True)
            #make month and day columns
            DataCompiled['Month'] = DataCompiled['datetime_bins'].dt.month
            DataCompiled['Day'] = DataCompiled['datetime_bins'].dt.day
        
        #Convert the start and end date to pandas series 
        StartDate = pd.Series(pd.to_datetime(Ranges['StartDate'].loc[i]))
        EndDate = pd.Series(pd.to_datetime(Ranges['EndDate'].loc[i]))
        
        #Make a mask of all entries with month and day after or equal to the start date
        m1 = ((DataCompiled['Month']>StartDate.dt.month[0]) | 
              ((DataCompiled['Day']>=StartDate.dt.day[0]) & (DataCompiled['Month']==StartDate.dt.month[0])))
        #Make a mask of all entries with month and day before or equal to the end date
        m2 = ((DataCompiled['Month']<EndDate.dt.month[0]) | 
              ((DataCompiled['Day']<EndDate.dt.day[0]) & (DataCompiled['Month']==EndDate.dt.month[0])))
       
        #Combine the two masks and use to grab the data in the date range
        mask = m1&m2
        #Mask out only the data we want to keep
        DataCompiled = DataCompiled[mask]
        #make a hr:min column
        DataCompiled['Time'] = DataCompiled['datetime_bins'].dt.strftime("%H:%M")
        
        #For each variable (plot)
        for ii in range(0,NumPlots):
            #grab the the hr:min column and the data column
            Rdata = pd.DataFrame({'Time':DataCompiled['Time'],'Data':DataCompiled[headers[ii]]})
            #set the dataframe index to the time column
            Rdata.set_index('Time')
            #Groupby the time column and generate descriptive statistics
            Rdata = Rdata.groupby('Time').describe()
            #Convert the index (the time column) to a datetime value
            Rdata.index = pd.to_datetime(Rdata.index)
            #Plot the mean value as the centerline
            axarr[ii].plot(Rdata.index,Rdata['Data']['mean'],color = colors[i],linewidth=3.0,label = Labels[i])
            #Plot lines at +/- 1 std deviation and fill between
            axarr[ii].plot(Rdata.index,Rdata['Data']['mean']+Rdata['Data']['std'],color = colors[i],linewidth=1.0,label='outer')
            axarr[ii].plot(Rdata.index,Rdata['Data']['mean']-Rdata['Data']['std'],color = colors[i],linewidth=1.0,label='outer')
            #Add shading between the +/- stddev lines
            axarr[ii].fill_between(Rdata.index,Rdata['Data']['mean']-Rdata['Data']['std'],Rdata['Data']['mean']+Rdata['Data']['std'],alpha = .5, color=colors[i],label='outer')
            
    #Format the plot xlabel to be date/time values        
    ticks = axarr[-1].get_xticks()
    StartTick = d.date2num(pd.to_datetime('00:00'))
    #Defind the end tick time
    EndTick = d.date2num(pd.to_datetime('00:00')+dt.timedelta(days=1))
    
    #For each of the plots, adjust the tick spacing to be in 1-hr increments
    for i in range(0,NumPlots):
        axarr[i].set_xticks(np.linspace(StartTick, EndTick, 5))
        #Set the x-tick minor ticks at 1-hr intervals
        axarr[i].set_xticks(np.linspace(StartTick, EndTick, 25), minor=True)
        #Format the labels
        axarr[i].xaxis.set_major_formatter(d.DateFormatter('%I:%M %p'))
    
    #Get lists of all handles and labels 
    handles, labels = axarr[-1].get_legend_handles_labels()
    #Mask off the labels that should not be included in the legend (basically anything not labeled 'outer')
    mask = [not x=='outer' for x in labels]
    #Find the index numbers of the mask values to keep
    m2 = np.where(mask)[0]
    #Generate a list of handles and labels
    la = [labels[index] for index in m2]
    #Generate a list of handles
    ha = [handles[index] for index in m2]
    
    #Add the legend to the figure
    f.legend(ha, la, loc='upper right',prop={'size': font_size})
    
    #Generate text for the figure title
    if Stations == 'all':
        text = Stations
    else:
        text = ''
        for x in Stations:
            text = text+str(x)+'_'
        text = text[:len(text)-1]
    
    #Replace with user-specified list of stations if one is provided
    if len(TitleOverride)>0:
        text = TitleOverride
    #Place the figure title
    f.suptitle('Station(s): '+text+'\n'+'mean +/- 1 stdev', fontsize=font_size)
    
    #Adjust the text size of each of the figure elements
    for i in range(0,NumPlots):
        for item in ([axarr[i].title, axarr[i].xaxis.label, axarr[i].yaxis.label] +
                 axarr[i].get_xticklabels() + axarr[i].get_yticklabels()):
            item.set_fontsize(font_size)
    #Save the figure
    f.savefig('Diurnals-Station_'+text+'.png')
    