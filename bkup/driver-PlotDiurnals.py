# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:29:39 2018

@author: MATTJE
"""

import PlotDiurnals as PlotD
import pandas as pd
import time

s1 = {}
e1 = {}
s1[0] = '1/1/2017'
e1[0] = '2/1/2017'
s1[1] = '6/1/2017'
e1[1] = '7/1/2017'

Ranges = pd.DataFrame({'StartDate':s1,'EndDate':e1})
Labels = {}
Labels[0] = 'Jan'
Labels[1] = 'Jun'
Readings = {}
Readings[0] = 'temp'
Readings[1] = 'solar'
Readings[2] = 'speed'
Readings[3] = 'dir'

#TitleOverride = ''
#for i in range(0,ALLdata.numFiles):
#    tic = time.time()
#    print('plotting diurnals for Station: '+str(i)+'/'+str(ALLdata.numFiles-1))
#    Stations=[i]
#    PlotD.PlotDiurnals(ALLdata,Ranges,Labels,Readings,Stations,TitleOverride)
#    toc = time.time()
#    print('    plotted in '+format(toc-tic,'.1f')+'sec')



Stations=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,39,40,41,42,43,44]
TitleOverride = 'AllExcept_12-35-37'
PlotD.PlotDiurnals(ALLdata,Ranges,Labels,Readings,Stations,TitleOverride)

