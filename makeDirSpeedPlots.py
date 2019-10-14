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
def MakeDirSpeedPlots(ALLdata,dist,font_size = 18):
    #initialize the figure
    fig, ax1 = plt.subplots(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    ax2 = ax1.twinx()
    
    #Make a dataframe containing the list of all timestamps in the global dataset, then find the min/max date
    #used to set the x-axis limits of the figure
    df = pd.DataFrame({'dates':[]})
    for i in range(0,ALLdata.numFiles):
        dt = pd.DataFrame({'dates':ALLdata.WSdata[i].data_binned['datetime_bins']})
        df = df.append(dt,ignore_index = True)
    minDate = df.min()
    maxDate = df.max()

    #k nearest neighbors to print
    maxNeigh = 3
    
    #size of moving window(timesteps)
    WindowSize = 288
    #Convert window size to timesteps
#    WindowSize = WindowSize*24*60/ALLdata.BinTimestep
    #Size 1/2 of the window rounded to the nearest integer (for grabbing the timestamp)
    HalfWindow = np.floor(WindowSize/2).astype(int)            
    
    
    #Loop over each station
    print('Printing plots of (dir1-dir2)*speed1*speed2')
    for i in range(0,ALLdata.numFiles):
        #Grab the data for Station 1
        data = ALLdata.WSdata[i].data_binned
        #Get the name of the current station
        name = ALLdata.WSdata[i].name
    
        #Loop over each nearest neighbor
        for x in range(0,maxNeigh):
            tic = time.time()
            #Print the current station so the user knows where the script is        
            print('Station: ' + str(i)+'/'+str(ALLdata.numFiles)+' | Neighbor: ' +str(x)+'/'+str(maxNeigh))
            
            
            #Extract an array of the distances from the current station to all other stations
            temp = np.array(dist[name])
            #Sort the distances
            temp.sort()
            #Grab the distance of the Xth nearest neighbor
            minDist = temp[x]
            
            #If the distance is valid, continue with plotting
            #NOTE: The distance matrix contains np.nan values for:
            #   1. The self-loop distance
            #   2. If the coordinates of one or both of the stations are unknown
            if ~np.isnan(minDist):
                #Find the index number of the Xth nearest neighbor
                NNidx = dist[name][dist[name]==minDist].index[0]
                #Grab the data and the name of the Xth nearest neighbor
                NNdata = ALLdata.WSdata[NNidx].data_binned
                NNname = ALLdata.WSdata[NNidx].name
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
                #Convert the direction delta and speed1 and speed2 to data frams
                t = pd.DataFrame({'val':dir_delta})
                s1 = pd.DataFrame({'val':data_merged['speed:_x']})
                s2 = pd.DataFrame({'val':data_merged['speed:_y']})
                #Multiply direction difference by speed1 and speed2
                Yvals = t*s1*s2
                
                breakhere=1                
                #Compile the differences into a dataframe
                delta = pd.DataFrame({'datetime':data_merged['datetime_bins'],'Yvals':Yvals['val']})
                

                #Make the title name string
                title = 'directiondifference-speed1-speed2 at '+ '(Station ID-'+str(i)+')' + 'NeighborNum(' + str(x) +')'
                #Plot the difference data as points
                lns1 = ax1.plot(delta['datetime'],delta['Yvals'],marker='.',ls='None',markerfacecolor='xkcd:blurple',label='Station'+str(i)+' vs. Station'+str(NNidx))
                #Get x-values for plotting the zero line
                axvals = [delta['datetime'].iloc[0],delta['datetime'].iloc[-1]]
                #Plot the zero marker line
                lns2 = ax1.plot(axvals,[0,0],color='xkcd:almost black',label = 'zero line')
                #Add figure title
                fig.suptitle('directiondifference*speed1*speed2' +'\n'+
                          'Station ID-'+str(i)+': '+' to Station ID-'+str(NNidx)+': '+'\n'+
                          'Nearest Neighbor: ' + str(x) + '\n' +
                          ' distance: ' + str(minDist.round(2)) + ' km', fontsize=font_size)
                #Set the x and y limits and label the axes
                ax1.set_ylim([-50,50])
                ax1.set_xlim(np.array([minDate,maxDate]))
                ax1.set_ylabel('Difference between station')
                ax1.set_xlabel('Date')
                
                ########
                #Prep for generating the variance moving window
                ########
                #Find the number of elements in the data range
                RawVals = np.array(delta['Yvals'])
                NumElem = RawVals.shape[0]
                #Calculate the number of windows
                NumWindows = NumElem-WindowSize+1
                
                #Preallocate for the averaged X and Y values
                varYvals = np.ones(NumWindows)*-999  #variance
                avgYvals = np.ones(NumWindows)*-999  #mean
                avgXvals = delta['datetime'].iloc[0:NumWindows]
                
                
                    
                #Slide moving window & grab time stamp from approx. middle of window range
                for c in range(0,NumWindows):
                    #Calculate the average over the window
                    avgYvals[c] = np.nanmean(RawVals[c:c+WindowSize])
                    #Calculate the variance over the window
                    varYvals[c] = np.var(RawVals[c:c+WindowSize])
                    #Find the index of the approx. central timestamp
                    xvind = c+HalfWindow
                    avgXvals[c] = delta['datetime'].iloc[xvind]
                
                #Plot the variance
                lns3 = ax2.plot(avgXvals,varYvals,color='xkcd:fire engine red',label = 'Variance over moving window')
                #Plot the mean
                lns3 = ax1.plot(avgXvals,avgYvals,color='xkcd:mango',label = 'Mean over moving window')
                
                #Center the y-axis limits around zero
                ylim = np.nanmax(varYvals)
                ax2.set_ylim([-1*ylim,ylim])
                #Add the legend
                fig.legend(prop={'size': font_size})
#                fig.legend()
                    
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
            