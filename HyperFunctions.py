# -*- coding: utf-8 -*-
"""
@author: Sonic
"""

import scipy.io as sio  
import numpy as np  
import matplotlib.pyplot as plt

def featureNormalize(X,type):
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
    elif type==3:
        maxX = np.max(X,0)
        X_norm = 2*X-maxX
        X_norm = X_norm/maxX
    return X_norm    

# def DrawResult(labels,imageID):
#     #ID=1:MUUFL
#     #ID=2:Indian Pines  
#     # ID=3: pavia  
#     #ID=7:Houston
#     #ID=8:Houston 18
#     global palette
#     global row
#     global col
#     num_class = int(labels.max())
#     if imageID == 1:
#         row = 326
#         col = 220
#         palette = np.array([[255,0,0],
#                             [0,255,0],
#                             [0,0,255],
#                             [255,255,0],
#                             [0,255,255],
#                             [255,0,255],
#                             [176,48,96],
#                             [46,139,87],
#                             [160,32,240],
#                             [255,127,80],
#                             [127,255,212]])
#         palette = palette*1.0/255
#     elif imageID == 2:
#         row = 145
#         col = 145
#         palette = np.array([[255,0,0],
#                             [0,255,0],
#                             [0,0,255],
#                             [255,255,0],
#                             [0,255,255],
#                             [255,0,255],
#                             [176,48,96],
#                             [46,139,87],
#                             [160,32,240],
#                             [255,127,80],
#                             [127,255,212],
#                             [218,112,214],
#                             [160,82,45],
#                             [127,255,0],
#                             [216,191,216],
#                             [238,0,0]])
#         palette = palette*1.0/255    
#     elif imageID == 3:
#         row = 610
#         col = 340
#         palette = np.array([[255,0,0],
#                             [0,255,0],
#                             [0,0,255],
#                             [255,255,0],
#                             [0,255,255],
#                             [255,0,255],
#                             [176,48,96],
#                             [46,139,87],
#                             [160,32,240]])
#         palette = palette*1.0/255  
#     elif imageID == 4:
#         row = 166
#         col = 600
#         palette = np.array([[255,0,0],
#                             [0,255,0],
#                             [0,0,255],
#                             [255,255,0],
#                             [0,255,255],
#                             [255,0,255]])
#         palette = palette*1.0/255  
#     elif imageID == 7:
#         row = 349
#         col = 1905
#         palette = np.array([[0, 205, 0],
#                             [127, 255, 0],
#                             [46, 139, 87],
#                             [0, 139, 0],
#                             [160, 82, 45],
#                             [0, 255, 255],
#                             [255, 255, 255],
#                             [216, 191, 216],
#                             [255, 0, 0],
#                             [139, 0, 0],
#                             [0, 0, 0],
#                             [255, 255, 0],
#                             [238, 154, 0],
#                             [85, 26, 139],
#                             [255, 127, 80]])
#         palette = palette*1.0/255
#     elif imageID == 8:
#         row = 601
#         col = 1192
#         palette = np.array([[0, 205, 0],
#                             [127, 255, 0],
#                             [46, 139, 87],
#                             [0, 139, 0],
#                             [160, 82, 45],
#                             [0, 255, 255],
#                             [255, 255, 255],
#                             [216, 191, 216],
#                             [255, 0, 0],
#                             [139, 0, 0],
#                             [0, 0, 0],
#                             [255, 255, 0],
#                             [238, 154, 0],
#                             [85, 26, 139],
#                             [255, 127, 80],
#                             [127,127,127],
#                             [85,85,85],
#                             [26,26,26],
#                             [46,46,46],
#                             [210,210,210]])
#         palette = palette*1.0/255
    
#     X_result = np.zeros((labels.shape[0],3))
#     for i in range(1,num_class+1):
#         X_result[np.where(labels==i),0] = palette[i-1,0]
#         X_result[np.where(labels==i),1] = palette[i-1,1]
#         X_result[np.where(labels==i),2] = palette[i-1,2]
    
#     X_result = np.reshape(X_result,(row,col,3))
#     plt.axis ( "off" ) 
#     plt.imshow(X_result)    
#     return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA