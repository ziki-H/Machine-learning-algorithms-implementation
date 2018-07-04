# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:30:59 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from sklearn import linear_model

def data_processed(train_data, train_label, test_data, test_label):
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    
    for i in range(0, len(train_label)):
        if(train_label[i] < 0):
            train_label[i] = 0
    for i in range(0, len(test_label)):
        if(test_label[i] < 0):
            test_label[i] = 0
            
    return (train_data, train_label, test_data, test_label)

#def logistic_regression(dataset, epochs, learning_rate):
#    feature_num = dataset.shape[1] - 1 #一共有几个feature,即数据集dataset的列数减去一,最后一列数据为label
#    dataset_num = dataset.shape[0]     #数据集的列数,即有多少样本数
#    data_X = dataset[:, 0:feature_num]
#    data_X = np.hstack( (np.ones((dataset_num, 1)), data_X) )
#    data_Y = dataset[:, feature_num]  #标签数据
#    weights = np.random.rand(feature_num+1, 1)  #logistic回归系数初始化
#    gradient_W = np.zeros((len(weights), 1))
#    for i in range(0, epochs):
#        WX = np.dot(data_X, weights)  
#        H_WX =  np.exp(WX) / ( 1 + np.exp(WX) ) #H_WX即预测的概率值输出
#        for i in range(0, len(gradient_W)):
#            gradient_W[i] = (1/dataset_num) * np.sum( (H_WX - data_Y) * data_X[:, i])
#        weights = weights - learning_rate * gradient_W
#        
#    return weights

def logistic_regression(train_data, train_label, epochs, learning_rate, decay):
    feature_num = train_data.shape[1]  #训练集一共有几个特征
    train_data_num = train_data.shape[0]  #一共有多少训练数据
    train_data = np.hstack( (np.ones((train_data_num, 1)), train_data ) ) #加上偏置项
    weights = np.ones((feature_num+1, 1))  #logistic回归系数初始化,初始化为[0,1)之间的随机数,符合均匀分布
    gradient_W = np.zeros((len(weights), 1))
    for i in range(0, epochs):
        WX = np.dot(train_data, weights)  #WX=W0 + W1*X1 + W2*X2.....
        H_WX =  np.exp(WX) / ( 1 + np.exp(WX) ) #H_WX即预测的概率值输出
        for j in range(0, len(gradient_W)):
            gradient_W[j] = ( 1 / (train_data_num)  * np.sum( (H_WX - train_label) * train_data[:, j]) )  #np.sum中表示矩阵对应元素相乘
        weights = weights - learning_rate * gradient_W
        learning_rate = learning_rate / (1 + decay*i)   #学习率随迭代步长衰减
        
        if(i % 10 == 0):
            error_num = 0
            error_rate = 0.0
            for k in range(0, len(H_WX)):
                if(abs( H_WX[k] - train_label[k] ) > 0.5):
                    error_num = error_num + 1
            error_rate = error_num / len(train_label)
            print("The error rate of training data is %.5f" %error_rate)
                
    return weights
  
if __name__ == '__main__':
    
    train_data = pd.read_excel("E:/HZQ/python/Machine Learning/logistics regression/diabetis_LR/diabetis_train_data.xlsx")
    train_label = pd.read_excel("E:/HZQ/python/Machine Learning/logistics regression/diabetis_LR/diabetis_train_label.xlsx")
    test_data = pd.read_excel("E:/HZQ/python/Machine Learning/logistics regression/diabetis_LR/diabetis_test_data.xlsx")
    test_label = pd.read_excel("E:/HZQ/python/Machine Learning/logistics regression/diabetis_LR/diabetis_test_label.xlsx")
    
    train_data, train_label, test_data, test_label = data_processed(train_data, train_label, test_data, test_label)
    
    epochs = 150
    learning_rate = 0.001
    decay = 0.01
    weights = logistic_regression(train_data, train_label, epochs, learning_rate, decay) 
    
    '''采用sklearn库中的logistic regression模型'''
    train_label = train_label.ravel()
    model = linear_model.LogisticRegression()
    model.fit(train_data, train_label)
    model_train_output = model.predict(train_data)
    error_num = 0
    error_rate = 0.0
    for k in range(0, len(model_train_output)):
        if( model_train_output[k] != train_label[k] ):
            error_num = error_num + 1
        error_rate = error_num / len(train_label)
    print("The error rate of training data is by sklearn is %.5f" %error_rate)
    
    
    
    

        
