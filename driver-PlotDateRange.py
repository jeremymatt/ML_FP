# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 08:33:04 2018

@author: MATTJE
"""

import PlotDataRange as PDR

StartDate = '12/19/2016'
EndDate = '12/22/2016'
StartDate = '3/30/2017'
EndDate = '4/2/2017'
Station = {}
Station[0] = 15
Station[1] = 22 
Station[2] = 23
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
#ReadingType[1] = 'speed'
Station = ['Station{:03d}'.format(Station[i]) for i in Station.keys()]
PDR.PlotDataRange(ALLdata,StartDate,EndDate,Station,ReadingType)