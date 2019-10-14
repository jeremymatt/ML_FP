# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:25:18 2019

@author: jmatt
"""
import numpy as np

#a1 = 345
#a2 = 10
#
#diff = abs(a1-a2)
#if diff>180:
#    diff = 360-diff
#    
#
#print('Difference: {}'.format(diff))
#
#
#diff = np.mod(a1-a2+180,360)-180
#print('Difference: {}'.format(diff))


station_sets = [[1,2,3],[4,5,6]]

l1 = [j for i in station_sets if i[0] !=1 for j in i]

l2 = []

for i in station_sets:
    if i[0]!=1:
        for j in i:
            l2.append(j)