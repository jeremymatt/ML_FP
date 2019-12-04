# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:38:44 2018

@author: MATTJE
"""


import matplotlib.pyplot as plt
from weather_loaddata import WEATHER_LOADDATA as LD


#Initialize:
ALLdata = LD()
#Load the raw data
destdir = './RawData/all/'
LoadType = 'raw'
#Load the binned data
destdir = './DataBinned/HRRR_INL-5minraw_noS35_no-HRRR/'
LoadType = 'binned'
ALLdata.LoadData(LoadType,destdir)
ALLdata.ConvertTimeStrToDatetime()
#ALLdata.Check_DST()

ALLdata.calc_wind_u_v(scale_by_speed=False)

label_file = 'data_labels_all.csv'
ALLdata.label_data(label_file)


num_samples = 1000
duration = '00:30:00'
start = '2017-01-01 00:00:00'
end = '2017-03-01 00:00:00'
variables = ['solar:','temp:','u','v']
stations = ALLdata.WSdata[:8]
samples, x_headers,y_headers = ALLdata.gen_multistep_samples(stations,start,end,num_samples,duration,variables)

mask = samples['sum_t1-n_labels']==0

samples.to_excel('samples_1k_30min.xlsx')


start = '2017-02-03 00:00:00'
end = '2017-02-06 00:00:00'
station = ALLdata.WSdata[4]
samples_test, x_headers,y_headers = ALLdata.gen_series_multistep_samples(station,start,end,duration,variables)


"""
num_timesteps = int(X.shape[1]/Y.shape[1])
for i in range(num_timesteps):
    start = 0+i*4
    end = 4+i*4
    print('start:{}, end:{}'.format(start,end))
    print(X[:1,start:end])
"""



"""

#MST_offset = 0
#ALLdata.ConvertTimeToUTC(MST_offset)



#Data is loaded into WSData objects with the following parameters:
ALLdata.WSdata[0].__dict__.keys()
ALLdata.WSdata[0].name
ALLdata.WSdata[0].data_binned.head(5)

#generate the distances adjancecy matrix
try: dist = pd.read_csv('distances_INL-only.csv').set_index('ID')
except: 
    FileName = 'stationcoords.csv'
    dist = gam.GEN_ADJ_MAT(ALLdata,FileName)
    file = open('distances.csv','w')
    dist.to_csv(file,mode='a',header=True,index=True,index_label='ID')
    file.close()
    
dist['keys']=dist.keys()
dist.set_index('keys',drop=True,inplace=True)

dist = ALLdata.trim_dist_matrix(dist)
    
    
    
day = 10
StartDate = '1/{}/2017'.format(day)
EndDate = '1/{}/2017'.format(day+1)
#EndDate = '3/1/2017'
#StartDate = '3/30/2017'
#EndDate = '4/2/2017'
Station = {}
Station[0] = 1
#Station[1] = 7 
#Station[2] = 8
#Station[3] = 19
#Station[4] = 24
#Station[5] = 30 
#Station[6] = 36 
#Station[7] = 21 
#Station[8] = 31 
#Station[9] = 34 
#Station = [32]
ReadingType = {}
ReadingType[0] = 'dir'
ReadingType[1] = 'speed'
#ReadingType[2] = 'temp'
Station = ['Station{:03d}'.format(Station[i]) for i in Station.keys()]
PDR.PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType)
    
run_descriptive_statistics = False
if run_descriptive_statistics:   
    var = 'solar:'
    data = pd.DataFrame(ALLdata.WSdata[15].data_binned[['datetime_bins',var]])
    data.set_index('datetime_bins',inplace=True)
    
    
    start = [2015,3,1]
    end = [2019,4,2]
    var = 'temp:'
    stations = ALLdata.WSdata
    station_QQ_plots(stations,start,end,var)
    
    
    for station in ALLdata.WSdata:
        station.data_binned.set_index('datetime_bins',inplace=True)
        
    print('test')
        
    times = []
    #times.append('2017-04-01 00:00:00')
    #times.append('2017-04-01 08:00:00')
    #times.append('2017-04-01 12:00:00')
    #times.append('2017-04-01 20:00:00')
    
    times.append('2017-01-01 00:00:00')
    times.append('2017-01-01 09:00:00')
    times.append('2017-01-01 12:00:00')
    times.append('2017-01-01 20:00:00')
    
    timestamps = [pd.to_datetime(time) for time in times]
    spatial_QQ_plots(ALLdata,timestamps)
    
    
    num_binned_readings = [station.data_binned.shape[0] for station in ALLdata.WSdata]
    plt.hist(num_binned_readings,bins=8)
    plt.xlabel('# data points per station')
    plt.ylabel('# stations')
               


#     
#
#if LoadType == 'raw':
#    ALLdata.CheckReadingRanges()
#    #Remove all records where all sensor values are zero
#    ALLdata.RemoveAllZeros()
#    #Check the time spacing between each reading in the data set
#    ALLdata.CheckTimeSpace()
#    
#    #Determine the minimum stepsize for which at least CutoffPercent of the 
#    #readings have an equal or smaller stepsize
#    CutoffPercent = 99
#    mask = ALLdata.TimeStepSum['% < or ==']>CutoffPercent
#    MinStepSize = np.array(ALLdata.TimeStepSum['StepSize(min)'][mask])[0]
#    
#    
#    timestep = 5 #minutes
#    options = 'avg'
#    firstYear = 2016 #the first year in the dataset
#    ALLdata.BinDataSets(timestep,options,firstYear)
#    print(ALLdata.BinTimestepSum)
#    ALLdata.SaveBinnedData()
    
    dirNormType = 'minmax'
    ALLdata.NormalizeVals('temp',dirNormType)
    ALLdata.NormalizeVals('speed',dirNormType)
    ALLdata.NormalizeVals('solar',dirNormType)
    
    
    import makeDirSpeedPlots as mDSP
    import makeDiffPlots as mDP
    mDSP.MakeDirSpeedPlots(ALLdata,dist)
    mDP.MakeDiffPlots(ALLdata,dist)
    
    
#    runfile('C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2/driver-PlotDiurnals.py', wdir='C:/Users/mattje/Documents/VMshare/BoiseBench/BoiseBenchGit/BB2')
    
"""