#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
boston = load_diabetes()
X = boston.data
y = boston.target
y.shape


# In[2]:


u = np.mean(X)
std = np.std(X)
X = (X-u)/std


# In[3]:


ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))


# In[4]:


def hypothesis(X,theta):
    y_ = np.dot(X,theta)
    return y_

def error(X,y,theta):
    m,n = X.shape
    y_ = hypothesis(X,theta)
    err = np.sum((y_-y)**2)
    return err/m

def gradient(X,y,theta):
    m,n = X.shape
    y_ = hypothesis(X,theta)
    grad = np.dot((y_-y).T,X)
    return grad/m

def gradientDescent(X, y, learning_rate = 0.1, epoch = 500):
    m,n = X.shape
    theta = np.zeros((n,))
    grad = np.zeros((n,))
    err = []
    for i in range(epoch):
        er = error(X,y,theta)
        err.append(er)
        grad = gradient(X,y,theta)
        theta = theta - learning_rate * grad
    return err,theta
    


# In[5]:


err,theta  = gradientDescent(X,y)


# In[6]:


plt.plot(err)


# In[7]:


def r2_score(y,ypred):
    ymean = y.mean()
    num = np.sum((y-ypred)**2)
    denum = np.sum((y-ymean)**2)
    score = 1 - num/denum
    return score


# In[8]:


ypred = hypothesis(X,theta)


# In[9]:


r2_score(y,ypred)


# In[11]:


print(ypred)


# In[ ]:




