#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 08:39:17 2017

@author: Yuliangze
"""

#Classifying HandWritten digits with TF.Learn

#All imports
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import tensorflow as tf
learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

#All downloaded images
mnist = learn.datasets.load_dataset('mnist')
#55,000 samples
data = mnist.train.images
labels = np.array(mnist.train.labels,dtype = np.int32)
#10,000 samples
test_data = mnist.test.images
test_labels = np.array(mnist.test.labels,dtype = np.int32)
'''
After you download all the images you need you will get:
    
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST-data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST-data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST-data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST-data/t10k-labels-idx1-ubyte.gz
'''

def display(i,trained = False):
    img = test_data[i]
    plt.title(f'Example: {i}, Label: {test_labels[i]}')
    #All those images are 28 by 28 grid cells
    plt.imshow(img.reshape((28,28)),cmap = plt.cm.gray_r)
    #print(len(data[0])) it will tell you there are 784 pixels
    #each pixel is a feature, so we have 784 features for each digit
    #And also the image has been platted 
    #It means the array of the image is actually a 1 D array
    #Thats why we need to reshaped the image to show it
  
#when you want to show more image, it will overwrite the plt data
#you can show it one by one
#display(1)
#display(2)

#initialize the classifier
feature_columns = learn.infer_real_valued_columns_from_input(data)
#we got 10 digits that why we have 10 classes
#And feature_columns is how many feature we use 784 by my guess
classifier = learn.LinearClassifier(n_classes = 10, feature_columns = feature_columns)
classifier.fit(data,labels,batch_size = 100, steps = 1000)
#out put will be LinearClassifier()

#All those Algorithm are inside the fit()
#evidence yi = sigma([Weight of each pixel] Wi,jXj)j
#Weights are adjusted by gradient descent
#It begins with random weights
#And as the time goes by. It will gradually adjusted to a better value.

#Evaluate the traing result
#we just need the piece of information of accuracy
#{'loss': 0.28329819, 'accuracy': 0.92159998, 'global_step': 1000}
accuracy = classifier.evaluate(test_data,test_labels)['accuracy']
print(accuracy)
#TIPS: we can easliy got 99% with Deep Learning

#look at all weights tension of each number
#positive weight will be drawn in red, negative weights will be drawn in blue
weights = classifier.get_variable_value("linear//weight/d/linear//weight/part_0/Ftrl_1")
f, axes = plt.subplots(2, 5, figsize=(10,4))
axes = axes.reshape(-1)
for i in range(len(axes)):
    a = axes[i]
    a.imshow(weights.T[i].reshape(28, 28), cmap=plt.cm.seismic)
    a.set_title(i)
    a.set_xticks(()) # ticks be gone
    a.set_yticks(())
plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



