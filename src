#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image

#creates a np array to store flattened images. since the size of each img is 28*28 pixels and we have 2400 training images, 
#the array is 2400 by 785 (one extra column because thats a numpy requiremet).
images = np.ones([2400,785])
path= "pathname"


#flattens all the trainin data
def getXTilda():
    path = "training folder's path name"
    X = np.ones([2400, 785 ])
    for i in range(2400):
        curri = i+1
        newpath = path + str(curri) + ".jpg"
        img = Image.open(newpath)
        image = np.asarray(img).flatten()
        image = np.append(image, [1])
    X[i] =  image
    return X

#get training labels
def labels(number):
    T = np.ones([2400])
    T *= -1
    start =  number*240
    for i in range (240):
        T[start+i] *=-1
    return T

#calculates first part of weight equation
def first_weight(X):
    Xtranspose = X.transpose()
    dotProduct = Xtranspose.dot(X)
    inverse = np.linalg.pinv(dotProduct)
    A = inverse.dot(Xtranspose)
    return A

def getWeight(A, Y):
    arr=[None] * 2400
    for i in range(2400):
        arr[i]=A*Y[i]
    return arr
#finding resulting labels. We will use this to compare the difference between
#result and actual labels
def test(path):
    X = getXTilda()
    A = getA(X)
    resultLabels = np.ones([200])
    
    for i in range(200):
        curri = i+1
        newpath = path + str(curri) + ".jpg"
        img = Image.open(newpath)
        image = np.asarray(img).flatten()
        image = np.append(image, [1])
    
        outputWeights = np.zeros([10])
        for j in range(10):
            currW = getW(firstWeight, labels(j)); 
            outputWeights[j] = image.dot(currW)
        maxElement = np.amax(outputWeights)
        index = np.where(outputWeights == maxElement)
        resultLabels[i] = index[0][0]
    return resultLabels

#building a confusion matrix to see difference in the two label sets
def confusionMatrix(original, result):
    cm = np.zeros([10, 10])
    for i in range(len(original)):
        cm[int(original[i])][int(result[i])] += 1
        
    return cm.astype(int)

resultLabels=test(path)


# In[ ]:


cm = confusionMatrix(originalLabels, resultLabels)
print(cm)


# In[ ]:




