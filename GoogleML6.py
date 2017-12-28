#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:15:44 2017

@author: Yuliangze
"""

#Train an image classifier
#with tensorflow for poets
#Only need to provide training data
#Deep Learning is a branch of machine learning with image learning
#Deep Learning is really good at classifying images
#you will use pixels inside each pictures to analyze
#and you dont need to get all the features of pictures manually

#In deep learning the classifier is called neural network

#Input to mutiple hidden layer to multiple hidden layers to output
#It can learn more complex data

#TF Learn(formly SK Flow)
#It is a high level ML library on the top of TensorFlow
#The syntax is similar to scikit-learn

#our task is to classify 5 types of flowers

#Tensorflow will not train a classifier from scrach
#it will get help from "Inception" an image classifier from google
#which is train from 1.2 Million pictures

#After that Tensorflow will do "retraining"
#so called "transfer learning"
#it will saves a lot of time and leverage prior work

#after you trained your classifier 
#you will only get result from what you trained like 5 types of flowers

#To train an classifier 
#1.Diversity of certain pics
#2.Quantity


from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib import learn

def main() :
    # Load dataset
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    
    # Build DNN with 10, 20, 10 units respectively
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)
 
    # Fit and predict
    classifier.fit(x_train, y_train, steps=200)
    prediciton = list(classifier.predict(x_test, as_iterable=True))
    score = metrics.accuracy_score(y_test, prediciton)

    print('Accuracy: {0:f}', format(score))

main()