# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:09:43 2018

@author: MATTJE
"""

import numpy as np
import pandas as pd
import copy as cp
from geopy.distance import vincenty


#Generates a matrix of distances from each station to every other station 
#included in the ALLdata object
def GEN_ADJ_MAT(ALLdata,FileName):
    #Import the station coordinates & build dataframe
    data = pd.read_csv(FileName,sep=',', header=0)
    data = pd.DataFrame(data)
    #Convert the names column into the row index
    data.set_index('Name',inplace=True)
    
    #Extract the number of stations & pre-allocate an array to store the distances
    numElem = ALLdata.numFiles
    distances = np.zeros([numElem,numElem])
    
    for i in range(0,numElem):
        #Grab the name of Station 1
        name1 = ALLdata.StationNames[i]
        #Grab the lat & lon values for the first station
        lat1 = data['lat'].loc[name1]
        lon1 = data['lon'].loc[name1]
        #Set the self-loop distance to 'not a number'
        distances[i,i] = np.NaN
        #Start loop at i+1 to halve the required number of iterations.  Allowable b/c the 
        #distance from A to B is assumed to be the same as from B to A
        for ii in range(i+1,numElem):
            #grab the name of the second station
            name2 = ALLdata.StationNames[ii]
            #Grab the lat/lon of the second tation
            lat2 = data['lat'].loc[name2]
            lon2 = data['lon'].loc[name2]
            #Check if the lat/lon of the two stations are known (-999 means unknown)
            if (lat1 == -999) | (lat2 == -999):
                #If the coordinates of either station are unknown, set the distance to 'not a number'
                #Set for both 'A to B' and 'B to A'
                distances[i,ii] = np.NaN
                distances[ii,i] = np.NaN
            else:
                #Calculate the vincenty distance between the two stations and 
                #store as kilometers
                dist = vincenty((lat1,lon1),(lat2,lon2)).kilometers
                #Set for both 'A to B' and 'B to A'
                distances[i,ii] = cp.copy(dist)
                distances[ii,i] = cp.copy(dist)
    
    #Make the distances into a dataframe
    dist = pd.DataFrame(distances)
    
    #Add the station names as column headers
    dist.columns = pd.Series(ALLdata.StationNames)
    
    #return the distance
    return dist