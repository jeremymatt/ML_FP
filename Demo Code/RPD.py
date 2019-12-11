# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:43:16 2019

@author: jmatt
"""

def RPD(v1,v2):
    """
    Returns the relative percent difference between two values:
    (v1-v2)/((v1+v2)/2), which is a measure of how dissimilar two values are.
    
    For strictly positive quanties, RPD ranges from 0% to 200%"""
    
    rpd = 100*(v1-v2)/((v1+v2)/2)
    
    return rpd