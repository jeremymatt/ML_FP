# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 12:52:48 2018

@author: MATTJE
"""


import pandas as pd
import numpy as np
import time
import copy as cp
import math
import numbers as num


class WEATHER_DATASET:
    def __init__(self):
        self.data = []
        self.BinTimeStep = 'NotBinned'
        self.BinOptions = 'NotBinned'
        breakhere=1
        
    def LoadRawData(self,fileName):       
        #####################
        #Code to open the new .eTWS files
        #####################       
            
        file = open(fileName)
        line = [file.readline()]
        ctr = 0
        self.name = 'NotFoundInFile'
        self.version = 'NotFoundInFile'
        self.height = 'NotFoundInFile'
        self.units = 'NotFoundInFile'
        while not line[ctr][:7]=='rec nr:':
            colPos = line[ctr].find(':')
            if line[ctr][:colPos+1]=='site name\t\t:':
                self.name = line[ctr][colPos+2:len(line[ctr])-1]
            if line[ctr][:colPos+1]=='version\t\t\t:':
                self.version =  line[ctr][colPos+2:len(line[ctr])-1]
            if line[ctr][:colPos+1]=='measurement height\t:':
                self.height =  line[ctr][colPos+2:len(line[ctr])-1]
            if line[ctr][:colPos+1]=='!units:':
                self.units =  line[ctr][colPos+1:len(line[ctr])-1]
            line.append(file.readline())
            ctr+=1
            
        file.close()
        
        #Read in the data from the header on down
        self.data = pd.read_csv(fileName,sep='\t', skiprows=ctr,header=0)
        
        startsize = self.data.shape[0]
        self.data.drop_duplicates(subset = ['year:','mon:','date:','hour:','min:','temp:','solar:','speed:','dir:'],inplace=True)
        self.data = self.data.reset_index(drop=True)
        self.NumDupsDropped = startsize -self.data.shape[0]
        
            
    def LoadBinnedData(self,fileName):
        #load pre-binned data
        file = open(fileName)
        line = [file.readline()]
        ctr = 0
        breakhere=1
        while not line[ctr][:5]=='index':
            colPos = line[ctr].find(':')
            line_name = line[ctr].split(':')[0].strip()
            if line_name=='StationName':
                self.name = line[ctr].split(':')[1].strip()
            if line_name=='version':
                self.version =  line[ctr].split(':')[1].strip()
            if line_name=='height':
                self.height =  line[ctr].split(':')[1].strip()
            if line_name=='BinTimeStep':
                self.BinTimeStep =  line[ctr].split(':')[1].strip()
            if line_name=='BinOptions':
                self.BinOptions =  line[ctr].split(':')[1].strip()
            if line_name=='units':
                self.units =  line[ctr].split(':')[1].strip()
            if line_name=='timezone':
                self.timezone = line[ctr].split(':')[1].strip()
            line.append(file.readline())
            ctr+=1
            breakhere=1
            
        file.close()
        
        #Read in the data from the header on down
        self.data_binned = pd.read_csv(fileName,sep=',', skiprows=ctr,header=0,index_col='index')
        breakhere=1
            
    def CheckReadingRanges(self,names,AllowableRange):
        temp = np.empty(4)
        temp[:] = np.nan
        self.NumOutsideRange = pd.DataFrame({'low':temp,'high':temp},index=names)
        
        for i in names:
            breakhere=1
            mask = self.data[i]>AllowableRange.loc[i]['high']
            if len(mask[mask])>0:
                breakhere=1
                self.data[i][mask] = np.nan
                self.NumOutsideRange.loc[i]['high']=len(mask[mask])
            mask = self.data[i]<AllowableRange.loc[i]['low']
            if len(mask[mask])>0:
                breakhere=1
                self.data[i][mask] = np.nan
                self.NumOutsideRange.loc[i]['low']=len(mask[mask])
            
            
            
        
            
       
#####################
#Code to open the original .tws files
#####################
        #Grab the first N lines and extract the version, name, and height info
#        N = 6
#        with open(fileName) as myfile:
#            head = [next(myfile) for x in range(0,N)]
#        
#        string = head[0]
#        self.version = string[string.find(':')+2:len(string)-1]
#        string = head[1]
#        self.name = string[string.find(':')+2:len(string)-1]
#        string = head[5]
#        self.height = string[string.find(':')+2:len(string)-1]
#        
#        self.NumReadings = self.data.shape[0]
        
    #Generates bins of size binMins and processes the sensor readings that fall into that bin
    #Can take options='first', 'last', or 'avg'
    def BinData(self,timestep,options,firstYear):
        #Check if the data was previously binned with the same settings
        SameLastSettings = (self.BinTimeStep==timestep) and (self.BinOptions == options)
        #If appropriate bins have been generated, don't re-create the bins
        if not SameLastSettings:
            #Store the bin settings for later comparison
            self.BinTimeStep = timestep
            self.BinOptions = options
            #Make date strings for use in the bin creation function (pd.date_range)
            startYr = '1/1/'+str(firstYear)
            lastYearVal = self.data['year:'].max()+1
            endYr = '1/1/'+str(lastYearVal)
                        
            #Warn the user if the start-year provided is later than the first year in the data
            if self.data['year:'].min()<firstYear:
                print('WARNING: FIRST DATE IN DATA FILE IS BEFORE 1/1/'+str(firstYear))
                print('     The first year in the dataset is: ' +str(self.data['year:'].min()))
            
            #Generate a list of bins to sort the data into
            BinList = pd.date_range(start=startYr,end=endYr,freq=str(timestep)+'min')
            #Convert to a pandas series
            MinuteBins = pd.Series(BinList)
            
            #Check if bins have already been created, and drop the column if they have
            m = self.data.keys() == 'datetime_bins'
            if len(m[m])>0:
                self.data.drop(columns='datetime_bins',inplace=True)
            
            
            #Extract the reading timestamps from the data dataframe
            x = pd.Series(self.data['datetime'])
            #Generate a series of bin values corresponding to the timestamps of the data
            temp = pd.cut(x,MinuteBins)
            binlist = pd.IntervalIndex(temp).left
            #Convert the bins to a dataframe and join it to the existing data dataframe
            y = pd.DataFrame({'datetime_bins':binlist})
            self.data = self.data.join(y)
        
        #Depending on the options selected, check for multiple readings in each bin
        #If more than one reading is found, either calculate the average or keep the first or last
        #Drop unecessary columns from the data dataframe
        if options == 'first':
            self.data_binned = self.data.drop_duplicates(subset = ['datetime_bins'],keep='first').reset_index()
            self.data_binned.drop(columns=['index','rec nr:','year:','mon:','date:','hour:','min:','datetime'],inplace=True)
        elif options == 'last':
            self.data_binned = self.data.drop_duplicates(subset = ['datetime_bins'],keep='last').reset_index()
            self.data_binned.drop(columns=['index','rec nr:','year:','mon:','date:','hour:','min:','datetime'],inplace=True)
        elif options == 'avg':
            self.data_binned = self.data.groupby('datetime_bins').mean().reset_index()
            self.data_binned.drop(columns=['rec nr:','year:','mon:','date:','hour:','min:'],inplace=True)
        else:
            print('INVALID OPTION: use one of first, last, avg')
        
        #Determine the max and min timesteps of the binned data
        dt = np.diff(self.data_binned['datetime_bins'])
        self.TimeStep = pd.DataFrame({'TimeStep':dt})
        self.TimeStep = self.TimeStep['TimeStep'].dt.seconds/60  
        self.MaxStep = self.TimeStep.max()
        self.MinStep = self.TimeStep.min()

        
#Iterative works faster on small data sets but is very slow on large data sets        
#        #using iterative method
#        print('iterative method')
#        tic = time.time()
#        
#        datetimes = pd.DataFrame({'datetime_bin2':self.data['datetime']})
#        self.data = self.data.join(datetimes)
#        
#        breakhere = 1
#        
#        for i in range(0,self.data.shape[0]):
#            dateval = self.data['datetime'].iloc[i]
#            mask = MinuteBins<=dateval
#            idx = mask[mask==True].index[-1]
#            binval = MinuteBins[idx]
#            self.data['datetime_bin2'].iloc[i] = binval
#            breakhere = 1
#        
#        toc = time.time()
#        print('Binned in ' +str(toc-tic)+'sec using iterative')    
        
        
    def RemoveAllZeros(self):
        #Generate a mask of locations where all sensor readings are zero
        b1 = self.data['solar:']==0
        b2 = self.data['temp:']==0
        b3 = self.data['speed:']==0
        b4 = self.data['dir:']==0
        mask = b1&b2&b3&b4
        #Find the indices of the all-zero values
        inds = self.data[mask==True].index
        #If there are records where all values are zero, remove those records and record the number of records dropped.
        if len(inds)>0:
            startlen = self.data.shape[0]
            self.data=self.data.drop(self.data.index[inds[0]:inds[-1]+1]).reset_index()
            self.RemovedZeros_all = startlen-self.data.shape[0]
        else:
            self.RemovedZeros_all = 0
        
    def RemoveStartingZeros(self):
        #Generate a mask of locations where all sensor readings are zero
        b1 = self.data['solar:']==0
        b2 = self.data['temp:']==0
        b3 = self.data['speed:']==0
        b4 = self.data['dir:']==0
        mask = b1&b2&b3&b4
        #Find the indices of the all-zero values
        inds = self.data[mask==True].index
        #If there are all-zero records, proceed to removal
        if len(inds)>1:
            #Calcluate the difference in inds[i] and inds[i+1] for all i in {0,len(inds)-1}
            stepsize = np.diff(inds)
            #If the the zero block starts at the first index and all the steps are 1,
            #then all of the all-zero records are at the start of the data
            if (inds.min()==0)&(stepsize.max()==1):
                #Set flag to report to user that there are no other all-zero records in the data
                self.OtherZeroBlocks = False
                #Number of records before removal of zeros
                startlen = self.data.shape[0]
                #Drop the all-zero records
                self.data2=self.data.drop(self.data.index[inds[0]:inds[-1]+1]).reset_index()
                #Calculate the number of records dropped.
                self.RemovedZeros = startlen-self.data2.shape[0]
            #If the zero block starts at the first index but the stepsize is 
            #greater than 1, then there are zeros in the middle of the data as well as at the start
            elif (inds.min()==0)&(stepsize.max()>1):
                #Let the user know that there are other zero blocks in the data
                self.OtherZeroBlocks = True
                #Find locations where the stepsize is greater than 1
                mask = stepsize>1
                #make a temporary copy of the index vector of N-1 elements to compare to the mask
                temp = cp.copy(inds[0:len(inds)-1])
                #Find the index of the first place where the stepsize is greater than 1
                #This is the index of the first record containing non-zero readings
                inds2 = temp[mask][0]
                #record the starting number of recrods
                startlen = self.data.shape[0]
                #Drop all-zero records
                self.data2=self.data.drop(self.data.index[inds[0]:inds2+1]).reset_index()
                #Record the number of records removed
                self.RemovedZeros = startlen-self.data2.shape[0]
            #If neither of the above cases are true, there are no zeros at the start but there
            #exists a block of all-zero values within the data
            else:
                self.RemovedZeros=0
                self.OtherZeroBlocks = True
        #No all-zero readings in the data
        else:
            self.RemovedZeros = 0
            self.OtherZeroBlocks = False
        

    #Locates duplicates timestamps with different sensor values
    #not really used because the dup. values are most likely a second reading
    #that occured within 1 minute.
    def FindDupEntries(self):
        #Make a boolean mask of duplicated values
        self.dups = self.data.duplicated(subset = ['year:','mon:','date:','hour:','min:'],keep=False)
        #Find the index of the duplicated values
        idx = self.dups[self.dups==True].index
        #Number of duplicated values
        self.NumHardDupEntries = idx.shape[0]
        
        if idx.shape[0] == 0:
            self.ContigDupBlock = 'NoDups'
            self.MaxEntriesPerMinute = 1
        else:
            i1 = cp.copy(idx[1:])
            i2 = cp.copy(idx[0:len(idx)-1])
            delta = i1-i2
            if delta.max()==1:
                self.ContigDupBlock = True
            else:
                self.ContigDupBlock = False
                
            self.MaxEntriesPerMinute = self.data[self.dups==True].groupby(['year:','mon:','date:','hour:','min:']).count()['dir:'].max()
        
        breakhere = 1
        
        

    #Returns the date/time variables for a specific entry 
    def DateVals(self,ind):
        breakhere = 1
        Vals = self.data.loc[self.data.index[ind],["year:","mon:","date:","hour:","min:"]].values.astype('int')
        return Vals

    #Returns the date/time variables for a specific entry 
    def DateVals_trimmed(self,ind):
        Vals = self.data_trimmed.loc[self.data_trimmed.index[ind],["year:","mon:","date:","hour:","min:"]].values.astype('int')
        return Vals

    #returns the weather values for a specific entry
    def WeatherVals(self,ind):
        Vals = self.data.loc[self.data.index[ind],["temp:","solar:","speed:","dir:"]].values.astype('int')
        return Vals
        
        
    #Check the number of minutes between readings
    def CheckSpacing(self):
        m = self.data.keys()=='datetime'
        if not len(m[m])>0:
            #Extract the time values
            DTnumbers = self.data[['year:','mon:','date:','hour:','min:']]
            #Convert the numbers to text strings
            dtsText = DTnumbers.applymap(str)
            #Convert to a dataframe and add dash, pipe, and colon columns for use separating the date/time numbers in the date/time string
            DTtemp = pd.DataFrame({'yr': dtsText["year:"],'mo': dtsText["mon:"],'da': dtsText["date:"],'hr': dtsText["hour:"],'mi': dtsText["min:"],'-': '-','|': '|',':': ':'})
            #Concatenate the date/time strings into a single-column dataframe 
            DTtext = pd.DataFrame({'datetime': DTtemp['yr']+DTtemp['-']+DTtemp['mo']+DTtemp['-']+DTtemp['da']+DTtemp['|']+DTtemp['hr']+DTtemp[':']+DTtemp['mi']})
            #Convert to datevalues
            self.datevals = pd.to_datetime(DTtext['datetime'],format='%Y-%m-%d|%H:%M')
            
            #Grab the set of t+1 timestamps
            later = np.array(self.datevals[1:])
            #Grab the set of t+0 timestamps
            earlier = np.array(self.datevals[:len(self.datevals)-1])
            #Subtract the t timestamps from the t+1 timestamps to get the timestep delta.  Store in a dataframe to allow use of the datetime accessor object
            delta = pd.DataFrame({'delta': later-earlier})
            #Convert from timedelta to minutes
            self.DeltaT = delta["delta"].dt.seconds/60        
            #Find the max and min time deltas
            self.MaxSpacing = self.DeltaT.max()
            self.MinSpacing = self.DeltaT.min()
            
            self.data = self.data.join(self.datevals)
#            temp = self.datevals.apply(lambda x: x.toordinal())
#            temp = pd.DataFrame({'timestamp':temp})
#            self.data = self.data.join(temp)
#            breakhere = 1
        
        
    #Converts the year,month,day,hour,minute values into a string that can be
    #read by np.datetime64  
    #TA is the output of the DateVals method
    def DateTimeStr(self,TA):
        yr = str(TA[0])
        mo = str(TA[1]).zfill(2)
        da = str(TA[2]).zfill(2)
        hr = str(TA[3]).zfill(2)
        mi = str(TA[4]).zfill(2)
        DStr = yr+'-'+mo+'-'+da+'T'+hr+':'+mi
        return DStr
    
#    #Adjusts the date/time values to multiples of 3 starting at zero (IE: 0,3,6, etc)
#    #This essentially 'bins' the readings into 3-minute intervals
#    def AdjTime(self):
#        #Extract the first minute in the entry
#        self.firstMin = self.data.iloc[0,5]
#        #Determine the adjustment (the modulus of the minute value)
#        self.timeADJ = (self.firstMin)%3
#        #Adjust the time values
#        self.data.iloc[:,5]-=self.timeADJ
#        #Adjust the firstMin variable
#        self.firstMin-=self.timeADJ
#        #Calculate how many readings are missing since the 0th minute
#        self.FirstIndex = math.floor(self.firstMin/3)
    
    
#    def TrimUseable(self,start,end):
#        #Trim the data to the useable block and reset the index to zero
#        self.data_trimmed = pd.DataFrame(self.data.iloc[start[0]:end[0],:],copy=True).reset_index(drop=True)
#        
#        
            
        
        
        