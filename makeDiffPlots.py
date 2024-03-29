# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:40:38 2018

@author: MATTJE
"""
####################################################
#OPERATION NOTES:
#   This is NOT a standalone script.  It requires that the ALLdata object has 
#   been loaded into memory and that the station adjacency matrix has been 
#   generated.  The following must be completed before running this:
#
#   1. From the driver.py script:
#        import numpy as np
#        import pandas as pd
#        import os
#        import time
#        import sklearn.cluster as cl
#        import matplotlib.pyplot as plt
#        from weather_loaddata import WEATHER_LOADDATA as LD
#        import pickle
#        import copy as cp
#        
#        ALLdata = LD()
#        
#        ALLdata.RemoveAllZeros()
#        
#        ALLdata.CheckTimeSpace()
#        
#        
#        CutoffPercent = 99
#        mask = ALLdata.TimeStepSum['% < or ==']>CutoffPercent
#        MinStepSize = np.array(ALLdata.TimeStepSum['StepSize(min)'][mask])[0]
#        
#        
#        timestep = 3 #minutes
#        options = 'avg'
#        firstYear = 2016 #the first year in the dataset
#        ALLdata.BinDataSets(timestep,options,firstYear)
#        print(ALLdata.BinTimestepSum)
#        #ALLdata.SaveBinnedData()
#        

#NOTE: Can be run without normalization, but plot limits will require adjustment
#        dirNormType = 'minmax'
#        ALLdata.NormalizeVals('temp',dirNormType)
#        ALLdata.NormalizeVals('speed',dirNormType)
#        ALLdata.NormalizeVals('solar',dirNormType)
#
#   2. Run gen_adj_mat.py
####################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import time

#requires the ALLdata object with temp, speed, and solar data normalized to {0:1}
def MakeDiffPlots(ALLdata,dist,font_size = 12):
    #initialize the figure
    fig, ax1 = plt.subplots(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')
    ax2 = ax1.twinx()
    
    #Make a dataframe containing the list of all timestamps in the global dataset, then find the min/max date
    #used to set the x-axis limits of the figure
    df = pd.DataFrame({'dates':[]})
    for i in range(0,ALLdata.numFiles):
        dt = pd.DataFrame({'dates':ALLdata.WSdata[i].data_binned['datetime_bins']})
        df = df.append(dt,ignore_index = True)
    minDate = df.min()
    maxDate = df.max()
    
    #Variable names
    names = {}
    names[0] = 'temp:'
    names[1] = 'solar:'
    names[2] = 'speed:'
    names[3] = 'dir:'
    names[4] = 'u'
    names[5] = 'v'
    
    
    #indices into names to print
    VariablesToCheck = np.array(list(names.keys()))
    
    #k nearest neighbors to print
    maxNeigh = 3
    #ranges for the plot y axis
    ranges = np.zeros([4,2])
    ranges[:,1] = [1,1,1,180]
    ranges[:,0] = [-1,-1,-1,-180]
    
    #size of moving window
    WindowSize = 288
    #Size 1/2 of the window rounded to the nearest integer (for grabbing the timestamp)
    HalfWindow = np.floor(WindowSize/2).astype(int)            
    
    
    #Loop over each station
    print('Printing difference/Variance plots')
    for i in range(0,ALLdata.numFiles):
        #Grab the data for Station 1
        data = ALLdata.WSdata[i].data_binned.copy()
        data.set_index('datetime_bins',drop=True,inplace=True)
        #Get the name of the current station
        name = ALLdata.WSdata[i].name
    
        #Loop over each nearest neighbor
        for x in range(0,maxNeigh):
            tic = time.time()
            #Print the current station so the user knows where the script is        
            print('Station: ' + str(i)+'/'+str(ALLdata.numFiles-1)+' | Neighbor: ' +str(x)+'/'+str(maxNeigh-1))
            
            
            #Extract an array of the distances from the current station to all other stations
            temp = dist[name]
            #Sort the distances
            temp = temp.sort_values()
            #Grab the distance of the Xth nearest neighbor
            minDist = temp.iloc[x]
            NNname = list(temp.index)[x]
            
            #If the distance is valid, continue with plotting
            #NOTE: The distance matrix contains np.nan values for:
            #   1. The self-loop distance
            #   2. If the coordinates of one or both of the stations are unknown
            if ~np.isnan(minDist):
                #Find the index number of the Xth nearest neighbor
                NNidx = ALLdata.StationNamesIDX[NNname]
                #Grab the data of the Xth nearest neighbor
                NNdata = ALLdata.WSdata[NNidx].data_binned.copy()
                #set the index to the datetime column
                NNdata.set_index('datetime_bins',drop=True,inplace=True)
                
                #Calculate the deltas
                delta = pd.DataFrame(data-NNdata)
                #Drop NaN rows (where one or the oher dataframe was missing data)
                delta.dropna(axis=0,inplace=True)
                #Reset the index
                delta.reset_index(inplace=True)
                
                delta['dir:'] = (delta['dir:']+180) % 360 - 180
                
                """
                #Merge the Station1 data with the data from the Xth nearest neighbor
                data_merged = pd.merge(data,NNdata,how='inner',left_on='datetime_bins',right_on='datetime_bins')
                
                #Calculate the temp, speed, solar, and direction differences
                temp_delta = data_merged['temp:_x']-data_merged['temp:_y']
                speed_delta = data_merged['speed:_x']-data_merged['speed:_y']
                solar_delta = data_merged['solar:_x']-data_merged['solar:_y']
                dir_delta = data_merged['dir:_x']-data_merged['dir:_y']
                #Mod the difference in direction values to find the min of the clockwise
                #and counterclockwise difference.  Positive is clockwise
                dir_delta = (dir_delta+180) % 360 - 180
                #Compile the differences into a dataframe
                delta = pd.DataFrame({'datetime':data_merged['datetime_bins'],'temp:':temp_delta,'speed:':speed_delta,'solar:':solar_delta,'dir:':dir_delta})
                """
                
                
                #For each reading type, generate a plot
                for ind, ii in enumerate(VariablesToCheck):
                    breakhere=1
                    #Make the title name string
                    title = names[ii] + ' at '+ '(Station ID-'+str(i)+')' + 'NeighborNum(' + str(x) +')'
                    #Plot the difference data as points
                    lns1 = ax1.plot(delta['datetime_bins'],delta[names[ii]],marker='.',ls='None',markerfacecolor='xkcd:blurple',label='Station'+str(i) + ' vs. Station'+str(NNidx))
                    #Get x-values for plotting the zero line
                    axvals = [delta['datetime_bins'].iloc[0],delta['datetime_bins'].iloc[-1]]
                    #Plot the zero marker line
                    lns2 = ax1.plot(axvals,[0,0],color='xkcd:almost black',label = 'zero line')
                    #Add figure title
                    fig.suptitle('Data Type: '+ names[ii] +'\n'+
                              'Station ID-'+str(i)+': '+' to Station ID-'+str(NNidx)+': '+'\n'+
                              'Nearest Neighbor: ' + str(x) + '\n' +
                              ' distance: ' + str(minDist.round(2)) + ' km', fontsize=font_size)
                    #Set the x and y limits and label the axes
                    ax1.set_ylim([ranges[ii,0],ranges[ii,1]])
                    ax1.set_xlim(np.array([minDate,maxDate]))
                    ax1.set_ylabel('Difference between station')
                    ax1.set_xlabel('Date')
                    
                    ########
                    #Prep for generating the variance moving window
                    ########
                    #Find the number of elements in the data range
                    RawVals = np.array(delta[names[ii]])
                    NumElem = RawVals.shape[0]
                    #Calculate the number of windows
                    NumWindows = NumElem-WindowSize+1
                    
                    #Preallocate for the averaged X and Y values
                    varYvals = np.ones(NumWindows)*-999  #variance
                    avgYvals = np.ones(NumWindows)*-999  #mean
                    avgXvals = delta['datetime_bins'].iloc[0:NumWindows]
                    
                    #Slide moving window & grab time stamp from approx. middle of window range
                    for c in range(0,NumWindows):
                        #Calculate the average over the window
                        avgYvals[c] = np.nanmean(RawVals[c:c+WindowSize])
                        #Calculate the variance over the window
                        varYvals[c] = np.var(RawVals[c:c+WindowSize])
                        #Find the index of the approx. central timestamp
                        xvind = c+HalfWindow
                        avgXvals[c] = delta['datetime_bins'].iloc[xvind]
                    
                    #Plot the variance
                    lns3 = ax2.plot(avgXvals,varYvals,color='xkcd:fire engine red',label = 'Variance over moving window')
                    #Plot the mean
                    lns3 = ax1.plot(avgXvals,avgYvals,color='xkcd:mango',label = 'Mean over moving window')
                    ax2.set_ylabel('Variance over a sliding window of ' +str(WindowSize)+' values')
                    #Center the y-axis limits around zero
                    ylim = np.nanmax(varYvals)
                    ax2.set_ylim([-1*ylim,ylim])
                    #Add the legend
                    fig.legend(prop={'size': font_size})
                    
                    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                             ax1.get_xticklabels() + ax1.get_yticklabels()):
                        item.set_fontsize(font_size)
                    
                    for item in ([ax2.title,  ax2.xaxis.label, ax2.yaxis.label] +
                             ax2.get_xticklabels() + ax2.get_yticklabels()):
                        item.set_fontsize(font_size)
                    
                    #Save the figure
                    plt.savefig(title+'.png')
                    #Clear the axes in preparation for the next figure
                    ax1.cla()
                    ax2.cla()

            
                
            #Stop the elapsed time tracker and print the elapsed time
            toc = time.time()
            print('      Plot generated in: ' +format(toc-tic,'.1f')+'sec')