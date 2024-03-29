# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 08:38:02 2018

@author: MATTJE
"""


import numbers as num
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
        
#Plots generates a plot of data between the start and end dates for the station or stations
#and the reading or readings requested
def PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType,label_prefix = None, font_size = 16):
    #Reading: one or more of the following:
    #   temp, solar, speed, dir
    #Station: one or more stations (either text name or ID#)
    
#        #Names of possible readings
#        names = ['temp','solar','dir','speed']
#        #Confirm that the user sent in a valid selection
#        if not ReadingType in names:
#            print('ERROR: Invalid reading type.  Reading must be one of:')
#            for i in names: print(i)
#            return
    
    #Initialize the figure
    fig, ax1 = plt.subplots(figsize=(22, 11), dpi= 80, facecolor='w', edgecolor='k')
    ax2 = ax1.twinx()
    hrs = mdates.HourLocator()
    minute = mdates.MinuteLocator(interval=15)
    hrs_fmt = mdates.DateFormatter('%H')
    
    #Convert the start and end dates to timestamp values
    StartDate = pd.to_datetime(StartDate)
    EndDate = pd.to_datetime(EndDate)
    
    #Generate list of station ID numbers
    StationNum=np.ones(len(Station)).astype(int)
    for i,val in enumerate(Station): 
        breakhere = 1
        #Get Station number if name is provided
        if not isinstance(Station[i],num.Number):
            StationNum[i] = ALLdata.StationNamesIDX[Station[i]]
        else:
            StationNum[i] = Station[i]
            
    NumPlots = len(Station)*len(ReadingType)
    colors = iter(cm.rainbow(np.linspace(0, 1, NumPlots)))
    
    
    #Grab the data for each reading type at each station and add to the plot
    for i in range(0,len(Station)):
        for reading in ReadingType.values():
            data = ALLdata.WSdata[StationNum[i]].data_binned
            m1 = data['datetime_bins']>=StartDate
            m2 = data['datetime_bins']<EndDate
            data = data.loc[m1&m2,:]
            
            plot_data = pd.DataFrame()
            plot_data['X'] = data['datetime_bins']
            plot_data['Y'] = data[reading]
            if label_prefix!=None:
                plot_data['label'] = data['{}|{}'.format(label_prefix,reading)]
            else:
                plot_data['label'] = 0
                
            #Build string for legend label
            DataLabel = ALLdata.StationNames[StationNum[i]]+'-'+reading
            
            mask_good = plot_data['label'] == 0
            #Plot the data
            cur_color = next(colors)
            if reading=='dir':
                ax2.plot(plot_data.loc[mask_good,'X'],plot_data.loc[mask_good,'Y'],label=DataLabel,marker='.',ls='None',color=cur_color)
                if sum(~mask_good)>0:
                    ax2.plot(plot_data.loc[~mask_good,'X'],plot_data.loc[~mask_good,'Y'],label='{}(fault)'.format(DataLabel),marker='*',markersize = 12, ls='None',color=cur_color)
            else:
                ax1.plot(plot_data.loc[mask_good,'X'],plot_data.loc[mask_good,'Y'],label=DataLabel,marker='.',ls='None',color=cur_color)
                if sum(~mask_good)>0:
                    ax1.plot(plot_data.loc[~mask_good,'X'],plot_data.loc[~mask_good,'Y'],label='{}(fault)'.format(DataLabel),marker='*',markersize = 12, ls='None',color=cur_color)
        
        
        """
        for ii in range(0,len(ReadingType)):
            breakhere=1
            #grab the relevant data
            temp = ALLdata.WSdata[StationNum[i]].data_binned[['datetime_bins',ReadingType[ii]+':']]
            #make a mask of the daterange
            mask = (temp['datetime_bins']>=StartDate)&(temp['datetime_bins']<EndDate)
            #Grab the plotdata
            PlotData_X = pd.DataFrame({'datetime_bins':temp['datetime_bins'][mask==True]})
            PlotData_Y = pd.DataFrame({ReadingType[ii]:temp[ReadingType[ii]+':'][mask==True]})
            
            if label_prefix!=None:
                
            #Build string for legend label
            DataLabel = ALLdata.StationNames[StationNum[i]]+'-'+ReadingType[ii]
            #Plot the data
            if ReadingType[ii]=='dir':
                ax2.plot(PlotData_X,PlotData_Y,label=DataLabel,marker='.',ls='None',color=next(colors))
            else:
                ax1.plot(PlotData_X,PlotData_Y,label=DataLabel,marker='.',ls='None',color=next(colors))
        """
    
    #Set the x-axis limits
    ax1.set_xlim(np.array([StartDate,EndDate]))
    fig.legend(prop={'size': font_size})
    #Figure Title string
    titletext = 'Date Range: '+chr(10)+StartDate.strftime('%Y-%m-%d %H:%M')+' to '+EndDate.strftime('%Y-%m-%d %H:%M')
    for i in range(0,len(ReadingType)):
        if not ReadingType[i] == 'dir':
            breakhere = 1
            titletext = titletext+chr(10)+ReadingType[i]


    fig.suptitle(titletext,fontsize=font_size)
    ax1.set_xlabel('date')
    ax1.set_ylabel('Normalized reading')
    ax2.set_ylabel('Direction')
    ax1.grid(which='major', axis='both',color='black')
    ax1.grid(which='minor', axis='both')
    ax1.xaxis.set_major_locator(hrs)
    ax1.xaxis.set_minor_locator(minute)
    ax1.xaxis.set_major_formatter(hrs_fmt)
    plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='center')
    
    
    
    
    #Resize the fonts for all items in axis 1
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
             ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(font_size)
    #Resize the fonts for all items in axis 2
    for item in ([ax2.title,  ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(font_size)
    
                