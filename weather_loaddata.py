# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Code to load and parse data from .tws weather station data
#Jeremy Matt
#2018-06-01

import numpy as np
import pandas as pd
import copy as cp
import os
import time
from weather_dataset import WEATHER_DATASET as WD
from IdahoDST_to_UTC import IdahoDST_to_UTC


class WEATHER_LOADDATA:
    def __init__(self):
        #Define the WSdata list
        self.WSdata = []
        
        
    def LoadData(self,LoadType,destdir):
        
        #Get the list of files in the data directory
        self.files = [f for f in os.listdir(destdir) if os.path.isfile(os.path.join(destdir,f))]
        
        #Determine the number of files to be processed
        self.numFiles = len(self.files)
        #Init the counter and pre-allocate space for starting dates and offsets
        ctr = 0
        self.NumEntries = np.zeros([self.numFiles,1]).astype('int')
        self.StationNamesIDX = {}
        self.StationNames = {}
        
        
        
        for F in self.files:
            #start elapsed time tracker
            tic = time.time()
            #generate the filename string
            fileName = destdir+F
            #Initialize the weather data object
            self.WSdata.append(WD())
            
            if LoadType == 'raw': 
                self.WSdata[ctr].LoadRawData(fileName)
                #Store the length of the each of the datasets
                self.NumEntries[ctr] = self.WSdata[ctr].data.shape[0]
            elif LoadType == 'binned':
                self.WSdata[ctr].LoadBinnedData(fileName)
            else:
                print('******ERROR******: Load type "'+LoadType+'" is not valid')
            
            
            self.StationNamesIDX[self.WSdata[ctr].name] = ctr
            self.StationNames[ctr] = self.WSdata[ctr].name
            #Increment the counter
            ctr+=1
            #Stop the elapsed time tracker and print the elapsed time
            toc = time.time()
            print('File: ' + str(ctr) + '/' +str(self.numFiles)+ ' loaded in: ' +format(toc-tic,'.1f')+'sec')
            


    def Check_DST(self):
        SummaryCompiled = {}
        
        TFsummary = pd.DataFrame({'2016':['NA'],'2017':['NA']})
        for i in range(1,45):
            TFsummary.loc[i] = ['NA','NA']
        
        DST2016 = pd.to_datetime('2016-03-13 1:59:00')
        DST2017 = pd.to_datetime('2017-03-12 1:59:00')
        
        
        for i in range(0,self.numFiles):
            data = self.WSdata[i].data['datetime']
            TimeDiff = np.diff(data).astype('timedelta64[m]').astype(int)
            #check for 1hr1min jumps (IE 1:59 to 3:00)
            mask = TimeDiff==61
            print('\n\nChecking DST for Station: '+str(i))
            
            
            if data.loc[0]<DST2016:
                TFsummary.loc[i]['2016'] = 'false'
            if data.loc[0]<DST2017:
                TFsummary.loc[i]['2017'] = 'false'
            
            
            if len(mask[mask]>0):
                print('    1hr1min jump summary:')
                TimeDiff_df = pd.DataFrame(TimeDiff)
                idx = TimeDiff_df[mask==True].index
                DispMat = pd.DataFrame()
                for ii in range(0,len(idx)):
                    ColName = 'Pair'+str(ii)
                    DispMat[ColName] = np.array([data.iloc[0],data.loc[idx[ii]],data.loc[idx[ii]+1],data.iloc[-1]])
#                    print(DispMat[ColName])
                    breakhere=1
                    
                    if data.loc[idx[ii]]==DST2016:
                        TFsummary.loc[i]['2016'] = 'TRUE'
                    if data.loc[idx[ii]]==DST2017:
                        TFsummary.loc[i]['2017'] = 'TRUE'
                    
                SummaryCompiled[i]=DispMat
                
            else:
                print('    No 1hr1min jumps')
                
                
            #check for jumps back in time(IE 1:59 to 1:00)
            mask = TimeDiff<0
            if len(mask[mask]>0):
                print('    Reverse jump summary:')
                TimeDiff_df = pd.DataFrame(TimeDiff)
                idx = TimeDiff_df[mask==True].index
                DispMat = pd.DataFrame()
                for ii in range(0,len(idx)):
                    ColName = 'Pair'+str(i)
                    DispMat[ColName] = data.loc[idx[ii]:idx[ii]+1]
                    print(DispMat[ColName])
                    
                
            else:
                print('    No Reverse jumps')
        breakhere=1
        return SummaryCompiled, TFsummary
    
    def ConvertTimeToUTC(self,MST_offset):
        #Generate a list of stations that are stored in memory
        station_list = [self.WSdata[x] for x in range(0,self.numFiles)]
        
        for ind,station in enumerate(station_list):
            #tell user where the process is
            if np.mod(ind,10)==0:
                print('Coverting Station {} of {} to UTC'.format(ind,len(station_list)-1))
            
            #If the station.timezone variable does not exist or is not equal 
            #to 'UTC', set the flag to convert the times to to UTC
            try: in_UTC = station.timezone == 'UTC'
            except: in_UTC=False
            
            #If the name starts with 'Station and the times are not in UTC
            #then convert to UTC
            if (station.name.split('n')[0] == 'Statio') &(not in_UTC):
                #Flag the data as being in UTC
                station.timezone = 'UTC'
                try: station.timezone_offset += MST_offset
                except: station.timezone_offset = MST_offset
                #Set the index to the datetime bins as req. by the Idaho_to_UTC
                #function
                df = station.data_binned.set_index('datetime_bins')
                #COnvert to UTC
                station.data_binned = IdahoDST_to_UTC(df,MST_offset)
                #reset the index name to the expected string
                station.data_binned.index.name='datetime_bins'
                
                #find and store the number of duplicates
                station.NumDupsDropped_UTC = len(station.data_binned.index)-len(station.data_binned.index.drop_duplicates())
                #Drop any duplicates that may have been created
                station.data_binned = station.data_binned.loc[~station.data_binned.index.duplicated()]
                #Reset the index to convert the datetime field back into a column
                station.data_binned.reset_index(inplace=True)
            else:
                #Set the timezone flag to UTC for files that are already in UTC
                station.timezone = 'UTC'
    
    def ConvertTimeStrToDatetime(self):
        #Generate a list of stations that are stored in memory
        station_list = [self.WSdata[x] for x in range(0,self.numFiles)]
        print(len(station_list))
        
        for ind,station in enumerate(station_list):
            #tell user where the process is
            if np.mod(ind,10)==0:
                print('Coverting Station {} of {} text timestamps to datetime objects'.format(ind,len(station_list)-1))
            
            #If the station timestamps are text, set flag to false
            in_datetime_object = True
            if type(station.data_binned['datetime_bins'][0]) == str:
                in_datetime_object = False
            
            #If the name starts with 'Station and the times are not in UTC
            #then convert to UTC
            if (station.name.split('n')[0] == 'Statio') &(not in_datetime_object):
                #Set the index to the datetime bins 
                station.data_binned.set_index('datetime_bins',inplace=True)
                #convert the index to datetimes
                station.data_binned.index=pd.to_datetime(station.data_binned.index)
                #reset the index name to the expected string
                station.data_binned.index.name='datetime_bins'
                
                station.data_binned.reset_index(inplace=True)
        
    def CheckReadingRanges(self):
        names = ['temp:','solar:','dir:','speed:']
        self.ErrorList = pd.DataFrame({'errors':['','','','']},index=names)
        ranges = ['high','low']
        temp = np.empty(4)
        temp[:] = np.nan
        AllowableRange = pd.DataFrame({'low':temp,'high':temp},index=names)
        self.RangeViolations = pd.DataFrame({'low':temp,'high':temp},index=names)
        AllowableRange['low'].loc['temp:'] = -150 #highest reliably recorded temp on earth is ~130F
        AllowableRange['high'].loc['temp:'] = 150 #lowest recorded temp on earth is -139F
        AllowableRange['low'].loc['solar:'] = 0
        AllowableRange['high'].loc['solar:'] = 1400 #highest TSI of 1361 recorded in upper atmosphere
        AllowableRange['low'].loc['dir:'] = 0
        AllowableRange['high'].loc['dir:'] = 360
        AllowableRange['low'].loc['speed:'] = 0
        AllowableRange['high'].loc['speed:'] = 260
        
        for x in range(0,self.numFiles):
            tic = time.time()
            self.WSdata[x].CheckReadingRanges(names,AllowableRange)
            
        
            for i in names:
                for ii in ranges:
                    prevCount = self.RangeViolations.loc[i][ii]
                    curCount = self.WSdata[x].NumOutsideRange.loc[i][ii]
                    self.RangeViolations.loc[i][ii] = np.nansum([prevCount,curCount])
                    if curCount>0:
                        self.ErrorList.loc[i] = self.ErrorList.loc[i]+str(x)+':'+i+'('+ii+')|'
                    
            toc = time.time()
            print('Checked reading ranges for file ' +str(x)+'/'+str(self.numFiles-1)+' in ' +format(toc-tic,'.1f')+'sec')
                    
        breakhere=1
        
    def SaveBinnedData(self):
        #Directory to save the binnned data in
        s_dir = './DataBinned/'
        for i in range(0,self.numFiles):
            #Tell User where script is in the process
            print('Saving file ' +str(i)+'/'+str(self.numFiles-1))
            #Generate the filename to save the data
            name = self.WSdata[i].name
            filename = s_dir+'BinType-'+self.DupHandling+'_BinTS-'+str(self.BinTimestep)+'_name-'+name+'.binned'
            #Open the file and write the header info
            file = open(filename,'w')
            file.write('StationName:'+name+chr(10))
            
            try: file.write('version:'+self.WSdata[i].version+chr(10))
            except: file.write('version:NotFoundInFile')
            
            try: file.write('height:'+self.WSdata[i].height+chr(10))
            except: file.write('height:NotFoundInFile')
            
            try: file.write('BinTimeStep:'+str(self.WSdata[i].BinTimeStep)+chr(10))
            except: file.write('BinTimeStep:NotFoundInFile')
            
            try: file.write('BinOptions:'+self.WSdata[i].BinOptions+chr(10))
            except: file.write('BinOptions:NotFoundInFile')
            
            try: file.write('timezone:'+self.WSdata[i].timezone)
            except: file.write('timezone:NotFoundInFile')
            
            try: file.write('units:'+self.WSdata[i].units+chr(10)+chr(10))
            except: file.write('units:NotFoundInFile')
            #Grab the data to be output
            ToOutput = self.WSdata[i].data_binned
            #flush() required prior to to_csv() for pandas version 0.23.1
            file.flush()
            #Write the data block
            ToOutput.to_csv(file,mode='a',header=True,index=True,index_label='index')
            #Close the file
            file.close()
            
        
            
    def FindDataRange(self,reading):
        #reading is one of "temp", "solar", "dir", "speed"
        #Names of possible readings
        names = ['temp','solar','dir','speed']
        #Init the min & max to false values
        Rmin = 9999
        Rmax = -9999
        #Confirm that the user sent in a valid selection
        if not reading in names:
            print('ERROR: Invalid reading type.  Reading must be one of:')
            for i in names: print(i)
        else:
            #For each file, check the min/max and compare to the previously encountered min/max
            for i in range(0,self.numFiles):
                tempmin = np.nanmin(self.WSdata[i].data_binned[reading+':'])
                tempmax = np.nanmax(self.WSdata[i].data_binned[reading+':'])
                breakhere=1
                if Rmin>tempmin:
                    Rmin = tempmin
                if Rmax<tempmax:
                    Rmax = tempmax
                    
            #Print the range for the user
            print('Range Summary for ' + reading +' ==> min: ' +str(Rmin)+' / max: ' + str(Rmax))
            
        #return the min/max values
        return Rmin, Rmax
                    
    
    def NormalizeVals(self,reading,dirNormType):
        #Names of possible readings
        names = ['temp','solar','dir','speed']
        #Confirm that the user sent in a valid selection
        if not reading in names:
            print('ERROR: Invalid reading type.  Reading must be one of:')
            for i in names: print(i)
            return
        
        if reading=='dir':
            #Select direction normalization type
            if dirNormType == 'minmax':
                self.NormalizeByMinMax(reading)
            else:
                #Figure out a better normalization method.......
                something=1
        else:
            self.NormalizeByMinMax(reading)
            
    def NormalizeByMinMax(self,reading):
        #Find the range of the values
        Rmin, Rmax = self.FindDataRange(reading)
        
        if not hasattr(self,'MinMaxRanges'):
            tempmin = np.array([999,999,999,-360])
            tempmax = -1*tempmin
            
            self.MinMaxRanges = pd.DataFrame({'Min':tempmin,'Max':tempmax},
                                        index=['temp','solar','speed','dir'])
            
        self.MinMaxRanges['Min'].loc[reading] = cp.copy(Rmin)
        breakhere = 1
        self.MinMaxRanges['Max'].loc[reading] = cp.copy(Rmax)
        
        breakhere=1
        
        for i in range(0,self.numFiles):
            #Extract the values from the binned dataframe
            breakhere=1
            vals = self.WSdata[i].data_binned[reading+':']
            #Overwrite the binned data with the normalized values
            self.WSdata[i].data_binned[reading+':'] = 1-(Rmax-vals)/(Rmax-Rmin)
            
        
    def RemoveAllZeros(self):
        #init a dict to store the number of zero records removed
        self.RemovedZeros_all = {}
        for i in range(0,self.numFiles):
            #Init timer
            tic = time.time()
            #Call the remove-zeros function for the current data set
            self.WSdata[i].RemoveAllZeros()
            #Store the number of removed zeros
            self.RemovedZeros_all[i] = self.WSdata[i].RemovedZeros_all
            #Stop the timer and print the elapsed time
            toc = time.time()
            print('Removed zero-readings from file ' +str(i+1)+'/'+str(self.numFiles)+'. RECORDS REMAINING: '+str(self.WSdata[i].data.shape[0]))
            #If all records were removed, warn the user (will cause subsequent operations to fail in a confusing manner)
            if self.WSdata[i].data.shape[0] == 0:
                print('*****************')
                print('  WARNING:')
                print('     ALL DATA POINTS REMOVED FROM FILE #:'+str(i))
                print('*****************')
        
            
    def BinDataSets(self,timestep,options,firstYear):
        #Store the duplicate handling method and timestep size (used to generate filenames for file output)
        self.DupHandling = options
        self.BinTimestep = timestep
        #Initizalize dicts to store the max and min step of the binned data for each file
        maxstep = {}
        minstep = {}
        #For each file, run the bin function, store the max & min timesteps, and report the elapsed time
        for i in range(0,self.numFiles):
            tic = time.time()
            self.WSdata[i].BinData(timestep,options,firstYear)
            maxstep[i] = self.WSdata[i].MaxStep
            minstep[i] = self.WSdata[i].MinStep
            toc = time.time()
            print('Binned file ' +str(i+1)+'/'+str(self.numFiles)+' in ' +format(toc-tic,'.1f')+'sec')
        #Join the max & min timesteps into a dataframe for easier viewing
        self.BinTimestepSum = pd.DataFrame({'MinStep(min)':minstep,'MaxStep(min)':maxstep})
            
    
    def CheckForDups(self):
        self.ContigDupBlocks = {}
        self.HardDupCounts = np.ones([self.numFiles,1]).astype('int')
        self.MaxEntriesPerMin  = np.ones([self.numFiles,1]).astype('int')
        ctr = 0
        for F in self.files:
            #Print the progress so the user knows where the script is
            print('Loading file: ' + str(ctr+1) + '/' +str(self.numFiles))
            #start elapsed time tracker
            tic = time.time()
            self.WSdata[ctr].FindDupEntries()
            
            #Collect information on duplicate entries
            self.ContigDupBlocks[ctr] = self.WSdata[ctr].ContigDupBlock
            self.HardDupCounts[ctr] = self.WSdata[ctr].NumHardDupEntries
            self.MaxEntriesPerMin[ctr] = self.WSdata[ctr].MaxEntriesPerMinute
            #Increment the counter
            ctr+=1
            #Stop the elapsed time tracker and print the elapsed time
            toc = time.time()
            print('    loaded in: ' +format(toc-tic,'.1f')+'sec')
            
    
   
    def CheckTimeSpace(self):
        #Preallocate space for the min & max spacings in the raw data (either before or after removal of all-zero values)
        self.minspacings = np.ones([self.numFiles,1]).astype('int')
        self.maxspacings = np.ones([self.numFiles,1]).astype('int')
        for ctr in range(0,self.numFiles):
            #start elapsed time tracker
            tic = time.time()
            #Call run the function to check spacing & record the values
            self.WSdata[ctr].CheckSpacing()
            self.minspacings[ctr] = self.WSdata[ctr].MinSpacing
            self.maxspacings[ctr] = self.WSdata[ctr].MaxSpacing
        
            #Stop the elapsed time tracker and print the elapsed time
            toc = time.time()
            print('Timestep for file ' + str(ctr+1) + '/' +str(self.numFiles)+ ' processed in: ' +format(toc-tic,'.1f')+'sec')
            
        #Initialize an empty dataframe
        df = pd.DataFrame({'StepSize(min)':[]})
        for i in range(0,self.numFiles):
            #Extract the stepsize values for file i
            dt = pd.DataFrame({'StepSize(min)':self.WSdata[i].DeltaT})
            #Append the stepsize values to the list of all other stepsize values
            df = df.append(dt,ignore_index = True)
        #Group like timestep values together and record the number of each timestep that exist in the data
        Summary = df.groupby('StepSize(min)').size()
        #Convert the unique timestep values to integers
        indVals = Summary.index.values.astype(int)
        #extract the count values from the summary dataframe to get rid of artifacts from the groupby function
        vals = np.array(Summary)
        #Calculate the cumulative percentage of timesteps that are less than or equal to a particular timestep
        CS = np.array(np.cumsum(vals)*100/len(df))
        #Store the timesteps and cumulative percentages in a dataframe
        self.TimeStepSum = pd.DataFrame({'StepSize(min)':np.array(indVals),'% < or ==':CS,'Raw Counts':vals})
        print()
        print('Summary of Timesteps in global data set:')
        print(self.TimeStepSum)
        print()
            
            
            
    def FindIndexOffset(self):
        #preallocate space for array of start indices
        self.StartUseableData = np.ones([self.numFiles,1])
        #loop through each file
        for i in range(0,self.numFiles):
            #Find the delta in minutes between the start of the current data stream and the start of the last data stream to begin
            self.StartUseableData[i] = (self.LatestStart-self.StartTimeStamp[i])/3
            
        #Calculate the length of the useable data stream from each station
        self.LenUseableData = np.subtract(self.NumEntries,self.StartUseableData)
        #Find the shortest useable length
        NumToUse = self.LenUseableData.min().astype('int')
        #Calculate the ending index of the useable block for each station
        self.EndUseableData = self.StartUseableData+NumToUse
        
        
    def TrimUseableData(self):
        #Pre-allocate space for the size of the useable block, the start times and the end times
        self.blocksize = np.ones([self.numFiles,1]).astype('int')
        self.StartTimes_trimmed = np.ones([self.numFiles,5]).astype('int')
        self.EndTimes_trimmed = np.ones([self.numFiles,5]).astype('int')
        #Loop through each file
        for i in range(0,self.numFiles):
            #Trim the data to the useable block and store in a separate dataframe
            self.WSdata[i].TrimUseable(self.StartUseableData[i].astype('int'),self.EndUseableData[i].astype('int'))
            #Store the block size, the start times, and the end times for QA inspection
            self.blocksize[i] = self.WSdata[i].data_trimmed.shape[0]
            self.StartTimes_trimmed[i] = self.WSdata[i].DateVals_trimmed(0)
            self.EndTimes_trimmed[i] = self.WSdata[i].DateVals_trimmed(-1)

    def GetAllData(self,type):
        colname = 'S1_'+type
        alldata = pd.DataFrame({colname: self.WSdata[0].data_trimmed[type]})
        for i in range(1,self.numFiles):
            colname = 'S'+str(i+1)+'_'+type
            temp = pd.DataFrame({colname: self.WSdata[i].data_trimmed[type]})
            alldata = alldata.join(temp)
        return alldata
        
    def GetStationNames(self):
        names = {}
        for i in range(0,self.numFiles):
            names[i] = self.WSdata[i].name
            
        self.all_names = pd.DataFrame({'names':names})
        
        
                
    def label_data(self,label_file,structure = 'data_binned'):
        """
        
        """
        
        data_labels = pd.read_csv(label_file)
        data_labels['start'] = pd.to_datetime(data_labels['start'])
        data_labels['end'] = pd.to_datetime(data_labels['end'])
        
        labeled_stations = set(data_labels['station'].values)
        labeled_params = set(data_labels['parameter'].values)
        
        for st_num in labeled_stations:
            station = 'Station'+str(st_num).zfill(3)
            station_index = self.StationNamesIDX[station]
            df = getattr(self.WSdata[station_index],structure)
            for param in labeled_params:
                header = 'label|{}'.format(param)
                df[header] = 0
            
            
#        for station in WSdata:
#            df = getattr(station,structure)
#            if not df.index.is_all_dates:
#                df.set_index(['datetime_bins'],inplace=True)
                
        
        for ind,row in data_labels.iterrows():
            station = 'Station'+str(row['station']).zfill(3)
            station_index = self.StationNamesIDX[station]
            df = getattr(self.WSdata[station_index],structure)
            
            mask1 = df['datetime_bins']>=row['start']
            mask2 = df['datetime_bins']<=row['end']
            header = 'label|{}'.format(row['parameter'])
            if not header in df.keys():
                df[header]=0
            
            df.loc[mask1&mask2,header]=1
            
            
    def calc_wind_u_v(self,scale_by_speed=False):
        """
        calculated the u and v components of the wind direction
        """
        
        if not scale_by_speed:
            rho=1
            
        for station in self.WSdata:
            print('Processing Station: {}'.format(station.name))
            if scale_by_speed:
                rho = np.array(station.data_binned['speed:'])
                
            phi = np.array(station.data_binned['dir:'])*2*np.pi/360
            v = rho * np.cos(phi)
            u = rho * np.sin(phi)
            
            station.data_binned['u'] = u
            station.data_binned['v'] = v
            
            l1 = 'label|speed:'
            l2 = 'label|dir:'
            test1 = l1 in station.data_binned.keys()
            test2 = l2 in station.data_binned.keys()
            
            if test1 & test2:
                uv_label = station.data_binned[l1]+station.data_binned[l2]
                uv_label = uv_label>0
                uv_label = uv_label.astype(int)
                station.data_binned['label|u'] = uv_label
                station.data_binned['label|v'] = uv_label
            
            
    def load_xyz(self,filename):
        """
        Adds northing/easting data to each station object
        """
        print('\nLoading x,y,z data')
        xyz = pd.read_excel(filename)
        xyz.set_index('Station',inplace=True)
        
        for station in self.WSdata:
            station.northing = xyz.loc[station.name,'northing']
            station.easting = xyz.loc[station.name,'easting']
            station.elevation = xyz.loc[station.name,'elevation']
            
    def get_krig_data(self,stations,start,end,binned=True,scaler_vars=None):
        """
        Extracts all parameter readings from the stations between the start
        and end time (inclusive of the start and end times).  if start==end, 
        only the values for a single time stamp are returned
        
        INPUTS:
            stations - a list of station objects (IE ALLdata.WSdata)
            start - start date/time string in the format "yyyy-mm-dd hh:mm:ss"
            end - end date/time string in the format "yyyy-mm-dd hh:mm:ss"
            variables - list of variables to include in the output
            binned - If true (default) extract binned data.  Otherwise extracts 
                    the raw data
            scaler - default (None) = no scaling.  Otherwise a tuple of:
                        1. A fitted scikit learn scaler
                        2. A list of variables to scale
                    
        OUTPUTS:
            data - pandas dataframe of the data with station name, northing, 
                    easting, elevation, and parameter readings
        """
        
        data = pd.DataFrame()
        
        
        for station in stations:
            if binned:
                station_data = station.data_binned.copy()
            else:
                station_data = station.data.copy()
                
            if scaler_vars != None:
                variables = scaler_vars[1]
                scaler = scaler_vars[0]
                station_data[variables] = scaler.transform(station_data[variables])
            
            #Generate a mask that matches the timeframe of interest
            m1 = station_data['datetime_bins']>=start
            m2 = station_data['datetime_bins']<=end
            #Add the data from the current station
            data_temp = pd.DataFrame(station_data[m1&m2])
            #Add the northing and eastings
            data_temp['northing'] = station.northing
            data_temp['easting'] = station.easting
            data_temp['elevation'] = station.elevation
            data_temp['station'] = station.name
            data = data.append(data_temp,sort=False)
        
        data.reset_index(drop=True,inplace=True)
        return data
    
    def trim_dist_matrix(self,dist):
        """
        Keeps only distances in the dist matrix 
        """
        
        data_stations = list(self.StationNamesIDX.keys())
        dist_stations = list(dist.keys())
        not_in_dist = [station for station in data_stations if not station in dist_stations]
        not_in_data = [station for station in dist_stations if not station in data_stations]
        
        if len(not_in_dist)>0:
            print('WARNING: the following stations are not in the distances matrix')
            print(not_in_dist)
            
        if len(not_in_data)>0:
            for station in not_in_data:
                print('removing station from distance matrix: {} '.format(station))
                dist.drop(labels=station, axis=0, inplace=True)
                dist.drop(labels=station, axis=1, inplace=True)
                
        return dist
    
    
    def gen_multistep_samples(self,stations,start,end,num_samples,duration,variables):
        """
        generates samples consisting of all entries in the "variables" list for 
        all input feature vectors that fall within (duration) minutes of a 
        randomly selected start time (t0).  Columns in the output pandas data 
        structure will consist of:
            1. time - the time associated with the start of the time block (t0)
            2. station - the station the sample was generated from
               the variable data, IE var1@t0, var2@t0, var1@t1, var2@t2, etc
            3. num_bad_readings - the count of the data flags for the block
                0 = no bad data
                1 = 1 bad reading, etc
                
        INPUTS
            stations - a list of station objects to process
            start - start time to collect samples from (t0>=start)
                    in the format 'yyyy-mm-dd hh:mm:ss'
            end - end time to collect samples from (t0+duration<=end) 
                    in the format 'yyyy-mm-dd hh:mm:ss'
            num_samples - the number of samples to generate
            duration - the time spread to collect samples over - the last time
                    step is <=t0+duration.  Must be in the format hh:mm:ss
                    or timedelta object
            variables - the variables to include in the sample
            
            
        OUTPUTS
            samples - pandas dataframe of the generated samples
            variable_headers - the headers of the data block
        """
        
        #Convert the start and end times to pandas datetime objects
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        #Convert the duration to a timedelta object)
        duration = pd.to_timedelta(duration)
        
        #move the start ahead so the the last part of the datablock cannot
        #be before the start date/time
        start = start+duration
        
        #Init an empty dataframe to hold the samples
        samples = pd.DataFrame()
        
        #loop through sample generation
        for i in range(num_samples):
            #tell user where the process is
            if np.mod(i,100)==0:
                print('Generating sample: {} of {}'.format(i,num_samples))
            
            #Randomly select a station
            station = np.random.choice(stations)
            #Mask of the input feature vectors in the valid time range
            m1 = station.data_binned['datetime_bins']>=start
            m2 = station.data_binned['datetime_bins']<=end
            #Extract a list of possible times to start the sample
            times = station.data_binned.loc[m1&m2,'datetime_bins']
            #Select the time to start the sample randomly
            sample_time = np.random.choice(times)
            #Generate the sample
            sample = self.gen_1_multistep_sample(station,sample_time,duration,variables)
            #Append to the dataframe
            samples = samples.append(sample,sort=False)
            
        #Extract the headers associated with the X variables 
        variable_headers = [var for var in samples.keys() if var.split('|')[0] in variables]
        y_headers = [key for key in variable_headers if int(key.split('|')[1]) == 0]
        x_headers = ['month','day','hour','minute']
        x_headers.extend([key for key in variable_headers if int(key.split('|')[1]) > 0])
        return samples, x_headers,y_headers
        
    
    
    def gen_series_multistep_samples(self,station,start,end,duration,variables):
        """
        generates one sample for each timestep between start and end.
        samples consist of all entries in the "variables" list for 
        all input feature vectors that fall within (duration) minutes of a 
        randomly selected start time (t0).  Columns in the output pandas data 
        structure will consist of:
            1. time - the time associated with the start of the time block (t0)
            2. station - the station the sample was generated from
               the variable data, IE var1@t0, var2@t0, var1@t1, var2@t2, etc
            3. num_bad_readings - the count of the data flags for the block
                0 = no bad data
                1 = 1 bad reading, etc
                
        INPUTS
            station - a single station object to process
            start - start time to collect samples from (t0>=start)
                    in the format 'yyyy-mm-dd hh:mm:ss'
            end - end time to collect samples from (t0+duration<=end) 
                    in the format 'yyyy-mm-dd hh:mm:ss'
            duration - the time spread to collect samples over - the last time
                    step is <=t0+duration.  Must be in the format hh:mm:ss
                    or timedelta object
            variables - the variables to include in the sample
            
            
        OUTPUTS
            samples - pandas dataframe of the generated samples
            variable_headers - the headers of the data block
        """
        
        #Convert the start and end times to pandas datetime objects
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        
        #Convert the duration to a timedelta object)
        duration = pd.to_timedelta(duration)
        
        
        #Mask of the input feature vectors in the valid time range
        m1 = station.data_binned['datetime_bins']>=start+duration
        m2 = station.data_binned['datetime_bins']<=end
        #Extract a list of possible times to start the sample
        times = station.data_binned.loc[m1&m2,'datetime_bins']
        
        #Init an empty dataframe to hold the samples
        samples = pd.DataFrame()
        
        #loop through sample generation
        for i,sample_time in enumerate(times):
            #tell user where the process is
            if np.mod(i,100)==0:
                print('Generating sample: {} of {}'.format(i,times.shape[0]))
            
            #Generate the sample
            sample = self.gen_1_multistep_sample(station,sample_time,duration,variables)
            #Append to the dataframe
            samples = samples.append(sample,sort=False)
            
        #Extract the headers associated with the X variables 
        variable_headers = [var for var in samples.keys() if var.split('|')[0] in variables]
        y_headers = [key for key in variable_headers if int(key.split('|')[1]) == 0]
        x_headers = ['month','day','hour','minute']
        x_headers.extend([key for key in variable_headers if int(key.split('|')[1]) > 0])
        return samples, x_headers,y_headers
        
        
        
    def gen_1_multistep_sample(self,station,sample_time,duration,variables):
        """
        generates samples consisting of all entries in the "variables" list for 
        all input feature vectors that fall within (duration) minutes of a 
        randomly selected start time (t0).  Columns in the output pandas data 
        structure will consist of:
            1. time - the time associated with the start of the time block (t0)
            2. station - the station the sample was generated from
               the variable data, IE var1@t0, var2@t0, var1@t1, var2@t2, etc
            3. num_bad_readings - the count of the data flags for the block
                0 = no bad data
                1 = 1 bad reading, etc
                
        INPUTS
            station - The station object to generate the sample from
            sample_time - The first sample time (t0) in the format 
                    'yyyy-mm-dd hh:mm:ss'
            duration - the time spread to collect samples over - the last time
                    step is <=t0+duration.  Must be in the format hh:mm:ss
                    or timedelta object
            variables - the variables to include in the sample
            
            
        OUTPUTS
            sample - A pandas dataframe containing 1 sample
        """
        
        #Convert the starting sample time to a datetime object
        sample_time = pd.to_datetime(sample_time)
        #Convert the duration to a timedelta object
        duration = pd.to_timedelta(duration)
        
        #Mask off the data in the time range
        m1 = station.data_binned['datetime_bins']>=sample_time-duration
        m2 = station.data_binned['datetime_bins']<=sample_time
        
        #Extract the data from the data table
        data = station.data_binned[m1&m2].copy()
        #Sort the data so the current data point is the first point
        data.sort_index(ascending=False,inplace=True)
        
        #Find the number of input feature vectors that fall inside the window
        num_rows = data.shape[0]
        
        #Init an empty dataframe
        sample = pd.DataFrame()
        #Add the time, station, day, hour, and minute to the dataframe
        sample['time'] = [data['datetime_bins'].iloc[0]]
        sample['time_min'] = [data]
        sample['station'] = [station.name]
        sample['month'] = [sample_time.month]
        sample['day'] = [sample_time.day]
        sample['hour'] = [sample_time.hour]
        sample['minute'] = [sample_time.minute]
        #Extract the variable values into a 1D vector
        vals = data[variables].values.ravel()
        #Generate headers to match the variables
        headers = ['{}|{}'.format(var,row) for row in range(num_rows) for var in variables]
        #Add the headers and values to the dataframe
        sample[headers] = pd.DataFrame([vals],index=sample.index)
        
        #Find the keys that are labels
        label_keys = [key for key in data.keys() if key.split('|')[0] == 'label']
        
        #Find the labels for the current time step
        t0_labels = data[label_keys].iloc[0].values
        #Add the current timestep labels into the dataframe
        sample[label_keys] = pd.DataFrame([t0_labels],index=sample.index)
        #Add the sum of all labels (t0 through tn) to the dataframe
        sample['sum_all_labels'] = data[label_keys].sum().sum()
        #Add the sum of all labels after t0 to the dataframe
        sample['sum_t1-n_labels'] = data[label_keys].iloc[1:].sum().sum()
        
        return sample
            
        
#    #Locates duplicates timestamps with different sensor values
#    #not really used because the dup. values are most likely a second reading
#    #that occured within 1 minute.        
#    def RemoveStartingZeros(self):
#        OtherZeroBlocks = {}
#        RemovedZeros = {}
#        for i in range(0,self.numFiles):
#            tic = time.time()
#            self.WSdata[i].RemoveStartingZeros()
#            OtherZeroBlocks[i] = self.WSdata[i].OtherZeroBlocks
#            RemovedZeros[i] = self.WSdata[i].RemovedZeros
#            toc = time.time()
#            print('Binned file ' +str(i)+'/'+str(self.numFiles)+'in ' +format(toc-tic,'.1f')+'sec')
#            
#        self.ZeroSummary = pd.DataFrame({'MidStreamBlocks':OtherZeroBlocks,'NumStartZerosRemoved':RemovedZeros})