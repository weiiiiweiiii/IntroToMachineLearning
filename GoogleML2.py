#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:49:04 2017

@author: Yuliangze
"""

#Many types of classifiers like
#artificial neural network
#support vector machine

#Iris Flower Data Set

#import data set
#train classifier
#predict label for new flower
#visualize tree
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

#Then split the data set into 2
#1. training data
#2. testing data

#this episode remove 1 example of each type of flower
test_idx = [0,50,100]

#training data set
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis = 0)
print(iris.data[1])
print(train_data[0])

#testing data
testing_target = iris.target[test_idx]
testing_data = iris.data[test_idx]

#classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#see the result
print(testing_target)
print(clf.predict(testing_data))

#viz code to visualize the tree
from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file = dot_data,
                     feature_names = iris.feature_names,
                     class_names = iris.target_names,
                     filled = True, rounded = True,
                     impurity = False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("iris.pdf")
