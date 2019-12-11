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
from RPD import RPD
import matplotlib as mpl


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler



#set plotting defaults
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.grid'] = True

filename = 'samples_20k_30min_prenorm.xlsx'
try: samples_all
except: samples_all = pd.read_excel(filename)

mask = samples_all['sum_t1-n_labels']==0
samples = samples_all[mask].copy()


variables = ['solar:','temp:','speed:','dir:','u','v']

#Extract the headers associated with the X variables 
variable_headers = [var for var in samples.keys() if var.split('|')[0] in variables]
y_headers = [key for key in variable_headers if int(key.split('|')[1]) == 0]
x_headers = ['month','day','hour','minute']
x_headers.extend([key for key in variable_headers if int(key.split('|')[1]) > 0])

label_headers = ['label|{}'.format(var) for var in variables]

X = samples[x_headers].values
Y = samples[y_headers].values
labels = samples[label_headers].values


y_scaler = StandardScaler()
y_scaler.fit(Y)
Y_scale = y_scaler.transform(Y)
Y_scale = Y

x_scaler = StandardScaler()
x_scaler.fit(X)
X_scale = x_scaler.transform(X)
X_scale = X

xtrain, xval, ytrain, yval,label_train,label_val = train_test_split(X_scale, Y_scale, labels, test_size=0.20, random_state=42)


 
X = samples_all[variable_headers].values
Y = samples_all[y_headers].values

X_scale = X
Y_scale = Y


labels = samples_all[label_headers].values
#
#Y_scale = y_scaler.transform(Y)
#allx_scaler =  StandardScaler()
#allx_scaler.fit(X)
#X_scale = allx_scaler.transform(X) 
#
#
#Y_scale = y_scaler.transform(Y)

#X_scale = X
#Y_scale = Y

xtrain, xval, ytrain, yval,label_train,label_val = train_test_split(X_scale, Y_scale, labels, test_size=0.20, random_state=42)
 
weight_fraction = labels.shape[0]/sum(labels)
    
direct_classifier = {}

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.figure(figsize=(10, 16))
epochs = [1000,1000,1000,1000,1000,1000]
for ind,var in enumerate(variables):
    ax = plt.subplot(3, 2, ind+1)
    plt.subplots_adjust(hspace=0.45,wspace=0.3)
    num_inputs = xtrain.shape[1]
    num_outputs = ytrain.shape[1]
    direct_classifier[var] = Sequential()
    direct_classifier[var].add(layers.Dense(100, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
    direct_classifier[var].add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
    direct_classifier[var].add(layers.Dense(1, kernel_initializer='normal',activation='sigmoid'))
    
    opt = 'adam'
    
    direct_classifier[var].compile(loss='binary_crossentropy', 
                     metrics = ['accuracy'],optimizer=opt)
    
    class_weight = {0:1,
            1:weight_fraction[ind]} 
    
    history = direct_classifier[var].fit(xtrain,label_train[:,ind],epochs=epochs[ind],
               class_weight = class_weight,batch_size=2000,
                     validation_data=(xval, label_val[:,ind]))
    
        
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('{} (best accuracy: {:0.3f}'.format(var,history.history['val_accuracy'][-1]))
    plt.ylim([0, 1])
    textstr = 'Best Validation Accuracy: {:0.3f}'.format(history.history['val_accuracy'][-1])
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top', bbox=props)



filename = 'Station008_samples_to_check.xlsx'
test_samples = pd.read_excel(filename)

#X_scale = allx_scaler.transform(test_samples[variable_headers])
X_scale = test_samples[variable_headers]
#ytest = y_scaler.transform(test_samples[y_headers])
labels = test_samples[label_headers].values


classification_val = {}
for ind,var in enumerate(variables):
    print('Classification Report for: {}'.format(var))

    classification_val[var] = direct_classifier[var].predict(X_scale)
    classification_val[var] = classification_val[var]>0.5
    print(classification_report(labels[:,ind],classification_val[var]))
    
    
for var in variables:
    filename = 'dir_class_model_{}.h5'.format(var)
    direct_classifier[var].save(filename)


from keras.models import load_model
for ind,var in enumerate(variables):
    filename = 'dir_class_model_{}.h5'.format(var)
    model = load_model(filename)
    cl = model.predict(X_scale)
    cl = cl>0.5
    print(classification_report(labels[:,ind],cl))
    
classes = pd.DataFrame()
prefix = 'pred_label'
classes['datetime_bins'] = test_samples['time']
for ind,var in enumerate(variables):
    classes['{}|{}'] = classification_val[var]


#
#prediction = model.predict(xtrain)
#
#pred_val = model.predict(xval)
#
#pred_unnorm = y_scaler.inverse_transform(prediction)
#
#rpd_unnorm = RPD(y_scaler.inverse_transform(ytrain),pred_unnorm)
#rpd_val = RPD(y_scaler.transform(yval),pred_val)
#
#
#plt.scatter(rpd_val.ravel(),label_val.ravel())
#
#for parameter in range(4):
#    plt.scatter(yval[:,parameter],y_scaler.inverse_transform(pred_val)[:,parameter])
#  
#classifier = Sequential()
#classifier.add(layers.Dense(50, input_dim=rpd_unnorm.shape[1], kernel_initializer='normal', activation='relu'))
#classifier.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
#classifier.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
#classifier.add(layers.Dense(4, kernel_initializer='normal', activation='sigmoid'))
#
#sgd = optimizers.Adam(learning_rate=0.05, decay=0.95)
#
#classifier.compile(loss='binary_crossentropy', 
#                    optimizer=sgd,
#                    metrics = ['accuracy'])
#
#
#classifier.fit(rpd_unnorm,label_train,epochs=20,batch_size=32,
#          validation_data=(rpd_val, label_val))
#
#
#classification_val = classifier.predict(rpd_val)
#classification_val = classification_val>0.5
#
#print('\nNumber of datapoints classified as faulty: {}'.format(sum(classification_val)))


#https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""
num_inputs = xtrain.shape[1]
num_outputs = ytrain.shape[1]
model = Sequential()
model.add(layers.Dense(100, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
model.add(layers.Dense(num_outputs, kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(xtrain,ytrain,epochs=20,batch_size=32,
          validation_data=(xval, yval))


regressions = {}
for ind,var in enumerate(variables):
    num_inputs = xtrain.shape[1]
    num_outputs = ytrain.shape[1]
    regressions[var] = Sequential()
    regressions[var].add(layers.Dense(100, input_dim=num_inputs, kernel_initializer='normal', activation='relu'))
    regressions[var].add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
    regressions[var].add(layers.Dense(1, kernel_initializer='normal'))
    
    regressions[var].compile(loss='mean_squared_error', optimizer='adam')
    
    regressions[var].fit(xtrain,ytrain[:,ind],epochs=500,batch_size=1000,
              validation_data=(xval, yval[:,ind]))


X = samples_all[x_headers].values
Y = samples_all[y_headers].values
labels = samples_all[label_headers].values

Y_scale = y_scaler.transform(Y)
X_scale = x_scaler.transform(X)

Y_scale = Y
X_scale = X

xtrain, xval, ytrain, yval,label_train,label_val = train_test_split(X_scale, Y_scale, labels, test_size=0.20, random_state=42)

pred = {}
rpd = {}
plt.figure(figsize=(6, 8))
for ind,var in enumerate(variables):
    pred[var] = regressions[var].predict(X_scale)
    rpd[var] = RPD(pred[var],np.array([Y_scale[:,ind]]).T)
    ax = plt.subplot(3, 2, ind+1)
    plt.subplots_adjust(hspace=0.45,wspace=0.3)
    plt.ylabel('Actual Reading')
    plt.xlabel('Predicted Reading')
    plt.scatter(pred[var],Y_scale[:,ind],label=var)
    plt.title(var)
 
plt.figure(figsize=(6, 8))
for ind,var in enumerate(variables):
    ax = plt.subplot(3, 2, ind+1)
    plt.subplots_adjust(hspace=0.55,wspace=0.4)
    plt.ylabel('Class')
    plt.xlabel('Normalized RPD')
    plt.scatter(rpd[var]/max(rpd[var]),labels[:,ind])
    plt.title(var)
    plt.title(var)
"""    
  


"""


filename = 'Station004_samples_to_check.xlsx'
test_samples = pd.read_excel(filename)

xtest = x_scaler.transform(test_samples[x_headers])
ytest = y_scaler.transform(test_samples[y_headers])






   

class_weight = {0:1,
                1:labels.shape[0]*labels.shape[1]/labels.sum()} 
    
classifiers = {}
for ind,var in enumerate(variables):
    
    classifiers[var] = Sequential()
    classifiers[var].add(layers.Dense(50, input_dim=rpd_unnorm.shape[1], kernel_initializer='normal', activation='relu'))
    classifiers[var].add(layers.Dense(50, kernel_initializer='normal', activation='relu'))
    classifiers[var].add(layers.Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    sgd = optimizers.Adam(learning_rate=0.005, decay=0.95)
    
    classifiers[var].compile(loss='binary_crossentropy', 
                        optimizer=sgd,
                        metrics = ['accuracy'])
    
    classifiers[var].fit(rpd_unnorm,label_train[:,ind],epochs=20,batch_size=32,
               class_weight = class_weight,
              validation_data=(rpd_val, label_val[:,ind]))
    
for ind,var in enumerate(variables):
    print('Classification Report for: {}'.format(var))

    classification_val = classifiers[var].predict(rpd_val)
    classification_val = classification_val>0.5
    print(classification_report(label_val[:,ind],classification_val))

"""


