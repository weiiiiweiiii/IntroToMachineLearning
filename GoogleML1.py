#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:22:08 2017

@author: Yuliangze
"""

#classifier  as a function
#data as input
#labels as output
#its known as supervised learn
#3 steps
#1. collect traing data
#2. train classifier
#3. make prediction

#measurement are called features

#classifier starts with a decision tree

from sklearn import tree
feature = [[140,1],[130,1],[150,0],[170,0]]
label = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf.fit(feature,label)
print(clf.predict([[160,0]]))