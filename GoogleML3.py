#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:10:53 2017

@author: Yuliangze
"""

#if we want to have a good classifier 
#we need good features which are useful to tell the difference

import numpy as np
import matplotlib.pyplot as plt

#data set numbers
greyhounds = 500
labs = 500

#average height (assumption) make then normally ditributed the height 
grey_height = 28+ 4*np.random.randn(greyhounds)
lab_height = 24+ 4*np.random.randn(labs)

plt.hist([grey_height,lab_height],stacked = True, color = ['r','b'])

#GOOD features would be
#1.informative
#2.independent
#3.simple

#for example if the eye color is not depend on the breed.
#the information of the eye color is useless