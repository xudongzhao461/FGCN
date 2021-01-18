import tensorflow as tf  
import numpy as np 
import math
import os

def CONV2D(x,name,shape,stride=[1,1,1,1],padding="SAME"):
    c_filter = tf.get_variable(name,shape,"float32",tf.truncated_normal_initializer())
    return tf.nn.conv2d(x,c_filter,stride,padding)

def generate_gabor_filter(shape,f=1,p=0.5):
    gama = math.sqrt(2)
    eta = math.sqrt(2)
    pi = math.pi
    N = shape[-1]
    d_x = shape[0]
    d_y = shape[1]
    x = np.linspace(1,d_x,d_x)
    y = np.linspace(1,d_y,d_y)
    [X,Y] = np.meshgrid(x,y)
    real_gabor_filterX = np.zeros([d_x,d_y,shape[-1]])
    real_gabor_filterY = np.zeros([d_x,d_y,shape[-1]])
    real_gabor_filter = np.zeros([d_x,d_y,shape[-1]])
    for i in range(N):
        theta = pi * (i) / N
        X_ = (X-(d_x+1)/2)*math.cos(theta) + (Y-(d_y+1)/2)*math.sin(theta)
        Y_ = -(X-(d_x+1)/2)*math.sin(theta) + (Y-(d_y+1)/2)*math.cos(theta)
        alpha = f/gama
        beta = f/eta
        A = f**2 / (pi*gama*eta)
        real_gabor_filterX[:,:,i] = A*np.exp(-(alpha**2 * X_**2 + beta**2 * Y_**2))*np.cos(-f*(X_)/math.sin(p)+(f*f+X_*X_)/(2*math.tan(p)))
        real_gabor_filterY[:,:,i] = np.cos(-f*(Y_)/math.sin(p)+(f*f+Y_*Y_)/(2*math.tan(p)))
        real_gabor_filter[:,:,i]=real_gabor_filterY[:,:,i]*real_gabor_filterX[:,:,i]
    g_f = np.zeros(shape)
    for i in range(N):
        g_f[:,:,:,i] = np.repeat(real_gabor_filter[:,:,i:i+1],shape[-2],axis=2)
    g_f = np.array(g_f)
    return g_f

def gen_gf_list(filter_size, f_list,p):
    gf_list = []
    for f in f_list:
        gf_list.append(generate_gabor_filter(filter_size, f,p))
    return gf_list

def filter_variable(name,shape,trainable=True):
    return tf.get_variable(name,shape,"float32",trainable=trainable,initializer=tf.truncated_normal_initializer())