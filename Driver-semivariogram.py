
# -*- coding: utf-8 -*-
#Author: Jeremy Matt
#Date: 9/16/19

import matplotlib as mpl
import numpy as np
import pandas as pd
import os
import time
import sklearn.cluster as cl
import matplotlib.pyplot as plt
from weather_loaddata import WEATHER_LOADDATA as LD
import pickle
import copy as cp
import gen_adj_mat as gam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import PlotDataRange as PDR
import scipy.stats as sps
from statsmodels.graphics.gofplots import qqplot
from select_daterange import select_daterange
from station_QQ_plots import station_QQ_plots
from spatial_QQ_plots import spatial_QQ_plots

#Import user-defined functions
from geostats_functions import *



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


#set plotting defaults
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.grid'] = True

dist = pd.read_csv('distances_INL-only.csv').set_index('ID')
dist['ks'] = list(dist.keys())
dist.set_index('ks',inplace=True)

datetime = '2017-04-01 12:00:00'

data = ALLdata.WSdata[0].data_binned[ALLdata.WSdata[0].data_binned['datetime_bins']==datetime]
ALLdata.calc_wind_u_v(scale_by_speed=False)

##make a list of the variable names to pass to functions
#variables = ['solar:','dir:']
#
##Generate a pandas dataframe of all pairs of data.  Note: both (i,j) and (j,i)
##pairs are included
##Set the function to return the unordered set of pairs
#unordered=True
#try: pairs
#except: pairs = extract_pairs(data,x,y,variables,unordered)

var = 'u'
head = []
tail = []
distances = []

#%% Select a subset of the stations in WSdata to make the semivariograms for a 
#smaller set of stations

station_list = ALLdata.WSdata
#station_list = ALLdata.WSdata[8:15]
#%%

for ind,station1 in enumerate(station_list):
    for station2 in station_list[ind+1:]:
        distances.append(dist.loc[station1.name,station2.name])
        data_head = station1.data_binned[station1.data_binned['datetime_bins']==datetime]
        data_tail = station2.data_binned[station2.data_binned['datetime_bins']==datetime]
        head.append(list(data_head[var])[0])
        tail.append(list(data_tail[var])[0])
        
        

pairs = pd.DataFrame({'dist':distances,'head|{}'.format(var):head,'tail|{}'.format(var):tail})

#Generate variable names to grab the correct columns from the pairs structure

sv_vars = ['head|{}'.format(var),'tail|{}'.format(var)]

#Calculate the pair-by-pair semivariance and store in a pandas dataframe
data_sv = SV_by_pair(pairs,sv_vars)

#Extract just the distance and semivariance columns from  the dataframe
differences = pd.DataFrame(data_sv[['dist','SV']])
#Rename the dist and SV columns to generic names to work with the semivariogram
#functions
differences.rename(columns={'dist':'distance','SV':'delta'},inplace=True)

#Labels for the x and y axes of the plots
labels = {}
labels['xlabel'] = '$\Delta$ distance (mm)'
labels['ylabel'] = 'semivariance'

#Set the binning mode type:
    #Bins of an equal distance width
    #The number of bins equal to the square root of the number of points
bins = [0,3,5,8,10,13,15,18,20,25,35,45,55,60,65,70,75,80,95,110,125,140,165]
sv_mode = {'bins':'sqrt','mode':'eq.dist'}
#sv_mode = {'bins':bins,'mode':'user.div'}

#Call the semivariogram generation function
x,y = generate_semivariogram(differences,labels,sv_mode)
N = differences.shape[0]

#Store the binned semivariogram data in a tuple to
sv_points = (x,y)



#set plotting defaults
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.grid'] = True

plt.figure()
plt.scatter(x,y)
plt.xlabel('distance')
plt.ylabel('semivariance')
plt.title('var')
#
##Generate an array of x-points at which to fit the models
#x_vals = np.linspace(0,max(x),100)
#
##The exponential model parameters
#sill = 6500
#rng = 30
#nugget = 1000
##Fit the exponential model
#model_y = fit_exponential(x_vals,sill,rng,nugget)
##Store the model x and y values
#model = (x_vals,model_y)
##Define the title and generate the plot
#title = 'exponential model fit\nN={}'.format(N)
#plot_model_fit(sv_points,model,labels,title)
#
#
#
##The gaussian model parameters
#sill = 6500
#rng = 20
#nugget = 1000
#model_y = fit_gaussian(x_vals,sill,rng,nugget)
##Store the model x and y values
#model = (x_vals,model_y)
##Define the title and generate the plot
#title = 'gaussian model fit\nN={}'.format(N)
#plot_model_fit(sv_points,model,labels,title)
#
##The spherical model parameters
#sill = 6500
#rng = 30
#nugget = 1000
##Fit the spherical model
#model_y = fit_spherical(x_vals,sill,rng,nugget)
##Store the model x and y values
#model = (x_vals,model_y)
##Define the title and generate the plot
#title = 'spherical model fit\nN={}'.format(N)
#plot_model_fit(sv_points,model,labels,title)
