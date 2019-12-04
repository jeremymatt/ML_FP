# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:07:26 2019

@author: jmatt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential 
from keras import layers
from keras import optimizers


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

filename = 'samples_10k_30min.xlsx'
samples = pd.read_excel(filename)

mask = samples['sum_t1-n_labels']==0

samples = samples[mask].copy()

variables = ['temp:','solar:','u','v']

#Extract the headers associated with the X variables 
variable_headers = [var for var in samples.keys() if var.split('|')[0] in variables]
y_headers = [key for key in variable_headers if int(key.split('|')[1]) == 0]
x_headers = ['month','day','hour','minute']
x_headers.extend([key for key in variable_headers if int(key.split('|')[1]) > 0])

X = samples[x_headers].values
Y = samples[y_headers].values


y_scaler = StandardScaler()
y_scaler.fit(Y)
Y_scale = y_scaler.transform(Y)
Y_scale = Y_scale

x_scaler = StandardScaler()
x_scaler.fit(X)
X_scale = x_scaler.transform(X)
X_scale = X_scale

xtrain, xval, ytrain, yval = train_test_split(X_scale, Y_scale, test_size=0.20, random_state=42)


#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

num_inputs = xtrain.shape[1]
num_outputs = ytrain.shape[1]
model = Sequential()
model.add(layers.Dense(100, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(num_outputs, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(xtrain,ytrain,epochs=20,batch_size=32,
          validation_data=(xval, yval))

model.predict()








