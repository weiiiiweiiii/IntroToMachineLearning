#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:54:53 2017

@author: Yuliangze
"""

#Own classifier from sracth
#k nearghest neigbors


#Eucdilean Distance
# d = (a^2+b^2+c^2+...)^-2
from scipy.spatial import distance

#Eucdilean Distance to get something like 
#d = ((a1-a2)^2+b(b1-b2)^2+(c1-c2)^2+...)^-2
def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN:

    def fit(self,X_train,y_train):
        #store the information we need
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self,X_test):
        #get out the prediction
        prediction = []
        #iter over the test rows
        for row in X_test:
            #get all the nearest neighbor(in this example there's only one)
            prediction.append(self.closest(row))
        return prediction
    
    def closest(self,row):
        #pick out a distance whatever it is
        #you can also use quick sort to find min
        best_dis = euc(row,self.X_train[0])
        best_index = 0
        #find the shortest distance between the point we want to test and 
        #all other points
        for i in range(1,len(self.X_train)):
            dis = euc(row,self.X_train[i])
            #get out the lowest distance from X_test between like 
            # [ 5.2  3.4  1.4  0.2](X_train) and  [ 6.7  3.3  5.7  2.5](X_test)
            if dis < best_dis:
                best_dis = dis
                best_index = i
        #every X_train has a value indicate to the y_train
        return self.y_train[best_index]



from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)


#from sklearn.neighbors import KNeighborsClassifier
#using our own classifier
clf = ScrappyKNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,predictions))