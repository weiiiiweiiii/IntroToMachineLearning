#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:06:43 2017

@author: Yuliangze
"""

#supervised learning
#pipline
#first we need to split two useing ways of data, train and test

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)


#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#we can try different classifiers to get the different result
#So we have so many different sophiscated classifiers 
#But in the high level, we got similar interfaces like .fit() and .predict()
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))

# we set up a model
#then to train the machine to adjust the parameters of the m and b like mx+b

#go tensorflow to playe with it !!!!