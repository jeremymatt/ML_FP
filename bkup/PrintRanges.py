# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:01:06 2018

@author: MATTJE
"""

import pandas as pd

names = ['temp:','solar:','speed:','dir:']

for i in range(0,ALLdata.numFiles):
    data = ALLdata.WSdata[i].data
    df = pd.DataFrame({'mins':data[names].min(),'maxs':data[names].max()},index=names)
    print('File '+str(i+1)+'/'+str(ALLdata.numFiles))
    print(df)
    if df.loc['solar:']['mins']<0 or df.loc['speed:']['mins']<0 or df.loc['dir:']['mins']<0:
        print('ERROR: invalid negative reading')
    print()
    