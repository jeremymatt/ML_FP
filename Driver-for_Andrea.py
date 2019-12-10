
# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19

import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from weather_loaddata import WEATHER_LOADDATA as LD
import gen_adj_mat as gam
import PlotDataRange as PDR
import geostats_functions as GSF
import predict_time_remaining as PTR
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler


#set plotting defaults
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.grid'] = True



#Initialize the data object:
ALLdata = LD()

#Location of the data to be loaded
destdir = './DataBinned/HRRR_INL-5minraw_noS35_no-HRRR/'
#load pre-binned data
LoadType = 'binned'

#Load the data
ALLdata.LoadData(LoadType,destdir)
#Convert the times to datetime objects
ALLdata.ConvertTimeStrToDatetime()

#Add variables for the U and V components of wind
ALLdata.calc_wind_u_v(scale_by_speed=True)

#Load the northing/easting data in to the data objects for each station
filename = 'northing_easting.xlsx'
ALLdata.load_xyz(filename)

#Specify the start and end times
start = '2017-01-01 11:00:00'
end = '2017-02-01 13:00:00'

start = pd.to_datetime(start)
end= pd.to_datetime(end)

start_hour = 11
end_hour = 12


data = ALLdata.WSdata[0].data_binned.copy()
data['hr'] = data['datetime_bins'].dt.hour

m1 = data['datetime_bins']>start
m2 = data['datetime_bins']<end
m3 = data['hr']>=start_hour
m4 = data['hr']<=end_hour



times_to_process = list(data.loc[m1&m2&m3&m4,'datetime_bins'].values)
times_to_process = data.loc[m1&m2&m3&m4,'datetime_bins']
num_times = len(times_to_process)


#Specify the list of stations to load
#ALLdata.WSdata is a list of objects that hold the data for each station
#Another valid call would be something like ALLdata.WSdata[:10], which would
#load the data for the first 10 stations
stations = ALLdata.WSdata 

x = 'easting'
y = 'northing'
variables = ['solar:','temp:','speed:','dir:','u','v']


generate = False
if generate:
    paired_data = pd.DataFrame()
    
    for ind,cur_time in enumerate(times_to_process):
        tic = time.time()
        #Generate a dataframe holding the data between the start and end times
        data = ALLdata.get_krig_data(stations,cur_time,cur_time)
        
        pairs = GSF.extract_pairs(data, x, y, variables)
        
        paired_data = paired_data.append(pairs)
        
        toc = time.time()
        
        
        PTR.predict(tic,toc,ind,num_times)
     
    half_length = int(paired_data.shape[0]/2)
    paired_data.reset_index(inplace=True,drop=True)
    paired_data.iloc[:half_length,:].to_excel('jan2017_midday_pairs_part1.xlsx')
    paired_data.iloc[half_length:,:].to_excel('jan2017_midday_pairs_part2.xlsx')

paired_data = pd.read_excel('jan2017_midday_pairs_part1.xlsx')
part2 = pd.read_excel('jan2017_midday_pairs_part2.xlsx')

paired_data = paired_data.append(part2)


#paired_data = pd.read_excel('jan2017_11AM_pairs.xlsx')
#%% Run the SV bin generation code
#Generate variable names to grab the correct columns from the pairs structure
primary = 'speed:'
variable = [primary,primary]
SV_vars = ['head|{}'.format(variable[0]),'tail|{}'.format(variable[1])]

data_sv = GSF.SV_by_pair(paired_data,SV_vars)

#Extract just the distance and semivariance columns from  the dataframe
differences = pd.DataFrame(data_sv[['dist','SV']])
#Rename the dist and SV columns to generic names to work with the semivariogram
#functions
differences.rename(columns={'dist':'distance','SV':'delta'},inplace=True)


#Labels for the x and y axes of the plots
labels = {}
labels['xlabel'] = '$\Delta$ Distance (km)'
labels['ylabel'] = 'Semivariance'

#Set the binning mode type:
    #Bins of an equal distance width
    #The number of bins equal to the square root of the number of points
    
bins = {}    
bins['temp:'] = [0.8063474494709624, 3.2060273344413943, 6.042012653042811, 8.659845254828738, 11.05952513979917, 14.113663175216082, 16.73149577700201, 18.69487022834145, 20.440091962865402, 29.166200635485154, 37.23785115765842, 44.65504352938522, 53.38115220200496, 59.70758098965428, 69.742605963167, 76.7234929012628, 81.95915810483464, 88.28558689248396, 94.17571024650229, 100.7202917509671, 105.30149880409249, 112.50053845900376, 118.82696724665308, 126.02600690156439, 132.5705884060292, 139.11516991049402, 144.56898783088135, 153.2950965035011, 158.96706714070393, 165.51164864516875]
 
bins['solar:'] = [0.8063474494709624, 3.2060273344413943, 6.042012653042811, 8.659845254828738, 11.05952513979917, 14.113663175216082, 16.73149577700201, 18.69487022834145, 20.440091962865402, 29.166200635485154, 37.23785115765842, 44.65504352938522, 53.38115220200496, 59.70758098965428, 69.742605963167, 76.7234929012628, 81.95915810483464, 88.28558689248396, 94.17571024650229, 100.7202917509671, 105.30149880409249, 112.50053845900376, 118.82696724665308, 126.02600690156439, 132.5705884060292, 139.11516991049402, 144.56898783088135, 153.2950965035011, 158.96706714070393, 165.51164864516875]

bins['u'] = [0.3865308314618119, 5.346897149008242, 9.22892296274022, 13.75795307876087, 17.424310791729972, 21.73767280698773, 25.619698620719717, 31.011401139791914, 34.246422651235235, 38.34411656573012, 44.1671552863281, 48.91185350311163, 54.08788792142094, 57.75424563439003, 64.65562485880245, 75.65469799770975, 88.16344784195725, 98.51551667857588, 109.51458981748318, 117.9256457472358, 125.68969737469978, 137.55144291665863, 150.49152896243191, 161.2749340005763, 161.6642280031448]
    
bins['v'] = [0.3865308314618119, 5.346897149008242, 9.22892296274022, 13.75795307876087, 17.424310791729972, 21.73767280698773, 25.619698620719717, 31.011401139791914, 34.246422651235235, 38.34411656573012, 44.1671552863281, 48.91185350311163, 54.08788792142094, 57.75424563439003, 64.65562485880245, 75.65469799770975, 88.16344784195725, 98.51551667857588, 109.51458981748318, 117.9256457472358, 125.68969737469978, 137.55144291665863, 150.49152896243191, 161.2749340005763, 161.6642280031448]

bins['speed:'] =  [1.4608055999174425, 2.9878746176258986, 4.73309635214985, 7.132776237120282, 9.314303405275218, 12.150288723876642, 14.331815892031578, 16.73149577700201, 19.567481095603426, 22.185313697389354, 25.239451732806266, 28.29358976822318, 30.9114223700091, 33.96556040542601, 35.92893485676546, 37.892309308104906, 40.29198919307533, 43.782432662123234, 45.527654396647186, 47.27287613117114, 48.799945148879594, 51.417777750665515, 53.16299948518947, 55.7808320869754, 57.744206538314835, 59.70758098965428, 61.67095544099373, 63.63432989233318, 68.21553694545855, 73.0148967153994, 81.74100538801915, 94.39386296331779, 105.30149880409249, 120.13588354754606, 134.97026829099963, 149.36834760082223, 161.80305245930538] 
  
sv_mode = {'bins':bins[primary],'mode':'user.div'}
#sv_mode = {'bins':'sqrt','mode':'eq.dist'}

#Call the semivariogram generation function
sv_output = GSF.generate_semivariogram(differences,labels,sv_mode,
                                       process_premade_bins=True,
                                       plot_raw = False)
N = differences.shape[0]

plt.figure()
plt.plot(sv_output['x'],sv_output['skew'])

plt.xlabel('distance (km)')
plt.ylabel('skew')
plt.title(primary)
plt.tight_layout()

bin_divs = pd.Series(sv_output['bins'])
bin_divs.to_excel('bins_mid-day_{}.xlsx'.format(primary.split(':')[0]))



#Generate an array of x-points at which to fit the models
x_vals = np.linspace(0,max(sv_output['x']),500)

#%% generate models
if primary == 'temp:':
    #Temperature Model parameters
    gram_type = 'SV'
    
    model = GSF.FIT_MULTI(gram_type)
    sill = 13
    rng = 60
    nugget = 5
    model.add_first_model(GSF.FIT_EXPONENTIAL,sill,rng,nugget)
    #Add the second model
    sill = 18
    rng = 125
    transition = 38
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    #Add the second model
    sill = 28
    rng = 155
    transition = 125
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    
    #Fit the exponential model
    model_y = model.sample(x_vals)
    #Store the model x and y values
    model_points = (x_vals,model_y)
    #Define the title and generate the plot
    title = '{} model fit of: {}\nN={:,}'.format(model.name,primary.split(':')[0],N)
    
    ylim = [0,'data']
#    ylim = 'data'
    GSF.plot_model_fit(sv_output,model_points,labels,title,ylim=ylim)
    
    
    
if primary == 'speed:':
    #Temperature Model parameters
    gram_type = 'SV'
    
    model = GSF.FIT_MULTI(gram_type)
    sill = 19
    rng = 30
    nugget = 5
    model.add_first_model(GSF.FIT_EXPONENTIAL,sill,rng,nugget)
    #Add the second model
    sill = 27
    rng = 80
    transition = 48
    model.add_next_model(GSF.FIT_EXPONENTIAL,sill,rng,transition)
    #Add the second model
    sill = 48
    rng = 155
    transition = 88
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    
    #Fit the exponential model
    model_y = model.sample(x_vals)
    #Store the model x and y values
    model_points = (x_vals,model_y)
    #Define the title and generate the plot
    title = '{} model fit of: {}\nN={:,}'.format(model.name,primary.split(':')[0],N)
    
    ylim = [0,'data']
#    ylim = 'data'
    GSF.plot_model_fit(sv_output,model_points,labels,title,ylim=ylim)
    


if primary == 'solar:':
    #Solar Model parameters
    gram_type = 'SV'
    
    model = GSF.FIT_MULTI(gram_type)
    sill = 4400
    rng = 35
    nugget = 750
    model.add_first_model(GSF.FIT_EXPONENTIAL,sill,rng,nugget)
    #Add the second model
    sill = 8000
    rng = 150
    transition = 50
    model.add_next_model(GSF.FIT_GAUSSIAN,sill,rng,transition)
    #Add the second model
    sill = 8000
    rng = 150
    transition = 60
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    
    #Fit the exponential model
    model_y = model.sample(x_vals)
    #Store the model x and y values
    model_points = (x_vals,model_y)
    #Define the title and generate the plot
    title = '{} model fit of: {}\nN={:,}'.format(model.name,primary.split(':')[0],N)
    
    ylim = [0,'data']
#    ylim = 'data'
    GSF.plot_model_fit(sv_output,model_points,labels,title,ylim=ylim)
    
    


if primary == 'u':
    #Wind u parameters
    gram_type = 'SV'
    
    model = GSF.FIT_MULTI(gram_type)
    #Add the first model
    sill = 20
    rng = 20
    nugget = 10
    model.add_first_model(GSF.FIT_LINEAR,sill,rng,nugget)
    #Add the second model
    sill = 55
    rng = 160
    transition = 48
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    
    #Fit the exponential model
    model_y = model.sample(x_vals)
    #Store the model x and y values
    model_points = (x_vals,model_y)
    #Define the title and generate the plot
    title = '{} model fit of: {}\nN={:,}'.format(model.name,primary.split(':')[0],N)
    
    ylim = [0,'data']
#    ylim = 'data'
    GSF.plot_model_fit(sv_output,model_points,labels,title,ylim=ylim)



if primary == 'v':
    #Wind v parameters
    gram_type = 'SV'
    
    model = GSF.FIT_MULTI(gram_type)
    sill = 37
    rng = 70
    nugget = 8
    model.add_first_model(GSF.FIT_LINEAR,sill,rng,nugget)
#    Add the second model
    sill = 51
    rng = 155
    transition = 104
    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
#    #Add the second model
#    sill = 28
#    rng = 155
#    transition = 125
#    model.add_next_model(GSF.FIT_LINEAR,sill,rng,transition)
    
    #Fit the exponential model
    model_y = model.sample(x_vals)
    #Store the model x and y values
    model_points = (x_vals,model_y)
    #Define the title and generate the plot
    title = '{} model fit of: {}\nN={:,}'.format(model.name,primary.split(':')[0],N)
    
    ylim = [0,'data']
#    ylim = 'data'
    GSF.plot_model_fit(sv_output,model_points,labels,title,ylim=ylim)


#%%
interp_points = pd.read_excel('interp_points_with_stations.xlsx')
target = list(zip(interp_points[x],interp_points[y]))

cur_time = '2017-01-01 12:00:00'

data_all = ALLdata.get_krig_data(stations,cur_time,cur_time)
data = pd.DataFrame()
data['val'] = data_all[primary]
data[x] = data_all[x]
data[y] = data_all[y]
data['station'] = data_all['station']

#Run the ordinary kriging algorithm on the test points
#sill = 13
#rng = 60
#nugget = 5
#model=GSF.FIT_EXPONENTIAL(sill,rng,nugget,gram_type)

OK = GSF.ORDINARY_KRIGING(model,data,x,y)
Z,EV,weights = OK.estimate(target)





interp_points[primary] = Z
interp_points['{}|EV'.format(primary)] = EV

line_to_plot = 'AD'
mask = interp_points['line']==line_to_plot
to_plot = interp_points.loc[mask,:]

labels = {}
labels['xlabel'] = 'Distance along line (km)'
labels['ylabel'] = '{}'.format(primary.split(':')[0])
labels['data_label'] = 'Kriged {}'.format(primary.split(':')[0])
labels['title'] = 'Kriged values along line {}'.format(line_to_plot)
labels['filename'] = 'kriged_{}_line-{}.png'.format(primary.split(':')[0],line_to_plot)
plot_x = 'line_dist'
plot_y = primary
plot_y2 = '{}|EV'.format(primary)

ax1,ax2 = GSF.plot_model_1D(to_plot,plot_x,plot_y,plot_y2,labels)


plt.figure()
station_mask = interp_points['type'] == 'station'
off_line_points = interp_points.loc[(~mask)&(~station_mask),:]
on_line_points = interp_points.loc[(mask)&(~station_mask),:]
off_line_stations = interp_points.loc[(~mask)&(station_mask),:]
on_line_stations = interp_points.loc[(mask)&(station_mask),:]
plt.plot(off_line_points['easting'],off_line_points['northing'],'k.',
         label = 'Points not on line {}'.format(line_to_plot))
plt.plot(on_line_points['easting'],on_line_points['northing'],'r.',
         label = 'Points on line {}'.format(line_to_plot))
plt.plot(off_line_stations['easting'],off_line_stations['northing'],'gP',markersize = 12,
         label = 'Points not on line {}'.format(line_to_plot))

plt.plot(on_line_stations['easting'],on_line_stations['northing'],'g*',markersize = 18,
         label = 'Points not on line {}'.format(line_to_plot))

plt.ylabel('Northing (km)')
plt.xlabel('Easting (km)')
plt.legend()


num_stations = data.shape[0]
errors = []
predicted = []
for i in range(num_stations):
    partial_data = data.loc[:i-1,:]
    partial_data = partial_data.append(data.loc[i+1:,:])
    partial_data = partial_data.copy()
    partial_data.reset_index(inplace=True,drop=True)
    target = [(data.loc[i,'easting'],data.loc[i,'northing'])]
    actual = data.loc[i,'val']
    OK = GSF.ORDINARY_KRIGING(model,partial_data,x,y)
    Z,EV,weights = OK.estimate(target)
    predicted.append(Z)
    errors.append(Z[0]-actual)

errors = np.array(errors)
MAE = np.abs(errors).mean()
ME = errors.mean()

plt.figure()

#set plotting defaults
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.grid'] = True

plt.plot(range(num_stations),errors,'k*')
plt.xticks(range(num_stations),data['station'],rotation = 'vertical')
plt.ylabel('Error (predicted-actual)')
plt.tight_layout()




    