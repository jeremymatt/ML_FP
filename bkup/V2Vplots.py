# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:26:05 2018

@author: MATTJE
"""

import numpy as np
import pandas as pd
import CompileAllData as CAD
import matplotlib.pyplot as plt

#Join all the data into a single dataframe
#defaults to outer join
JoinedData = CAD.CompileAllData(ALLdata)
#Directory to save the plots in 
OutputDir = './Plots/New/'
font_size = 18
#Init the figure axes
fig, ax1 = plt.subplots(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
#Find the names of the variables in the joined dataframe
Variables = JoinedData.keys()
#Number of variables to process
NumVars = len(Variables)
#For each variable
for i in range(0,NumVars):
    #Extract the name of the current x-axis variable
    Var1 = Variables[i]
    #Get the current x-axis data
    x_vals = JoinedData[Var1]
    #For each variable that hasn't been plotted on the x-axis
    for ii in range(i+1,NumVars):
        #Tell the user where the script is
        print('Plotting X_variable: '+str(i)+'/'+str(NumVars-1)+' and Y_variable: '+ str(ii-i-1)+'/'+str(NumVars-i-1))
        #Extract the name of the y-axis variable
        Var2 = Variables[ii]
        #get the current y-axis data
        y_vals = JoinedData[Var2]
        #Plot the data
        ax1.plot(x_vals,y_vals,marker='.',ls='None')
        
        #Remove the trailing colon from the variable name
        name1 = Var1[:len(Var1)-1]
        name2 = Var2[:len(Var2)-1]
        #Set the plot axes names
        ax1.set_xlabel(name1)
        ax1.set_ylabel(name2)
        #Find the station/variable type separator
        b1 = name1.find('_')
        b2 = name2.find('_')
        #Extract Station IDs and parameter types
        ID1 = name1[:b1]
        Par1 = name1[b1+1:]
        ID2 = name2[:b2]
        Par2 = name2[b2+1:]
        #Generate the title to group parameter types over station names
        title = Par1+'_'+ID1 + '-' + Par2+'_'+ID2
        
        #Update the font sizes
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                 ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(font_size)
        
        #Save the figure and clear the axes
        plt.savefig(OutputDir+title+'.png')
        ax1.cla()
        