# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:01:20 2019

@author: jmatt
"""

def predict(start,end,completed_steps,total_steps):
    """
    Predicts the remaining time based on the time to complete the most recent
    iteration and the remaining number of iterations
    
    INPUTS
        start - the starting timestamp from time.time()
        end - the ending timestamp
        completed_steps - the number of completed steps
        total_steps - the total number of iterations
        
    """
    #The time to complete the current step
    cur_dt = end-start
    
    #Calculate the number of remaining steps
    remaining_steps = total_steps-completed_steps
    
    #Estimated time to completion in seconds
    dt_sec = cur_dt*remaining_steps
    
    #Calculate hours to completion
    dt_hrs = dt_sec/(3600)
    hrs = int(dt_hrs)
    #Calculate minutes to completion
    dt_mins = (dt_hrs-hrs)*60
    mins = int(dt_mins)
    #calculate seconds to completion
    sec = int(60*(dt_mins-mins))
    
    #If iteration too more than 1 minute, report time in mins    
    if cur_dt>60:
        cur_dt = cur_dt/60
        units = 'mins'
    else:
        units = 'secs'
    
    #Print the string
    print('Completed iteration {}/{} in {:0.2f} {} (Est. time remaining: {}:{}:{})'.format(completed_steps,
          total_steps,
          cur_dt,
          units,
          str(hrs).zfill(2),
          str(mins).zfill(2),
          str(sec).zfill(2)))