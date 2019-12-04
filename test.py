# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:31:40 2019

@author: jmatt
"""

import pandas as pd
import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])
z = np.array([7,8,9])

test = pd.DataFrame({'X':x,'Y':y,'Z':z})
test.values.ravel()
test
test.loc[0:1,['X','Z']].values.T.ravel()
