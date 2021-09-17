import argparse
import time
import sys
import os
from HyperFunctions import *
from opsfrac import *
import tensorflow as tf
from tf_octConv import *

parser = argparse.ArgumentParser()
parser.add_argument('--conv_size',
                    type=int,
                    default=15,
                    help='size of kernel')
parser.add_argument('--order',
                    type=int,
                    default=25,
                    help='fractional order')
parser.add_argument('--lr',
                    type=int,
                    default=100,
                    help='learning rate')
parser.add_argument('--alpha',
                    type=int,
                    default=2,
                    help='alpha of octave')
parser.add_argument('--numsample',
                    type=int,
                    default=100,
                    help='number of training samples')


args = parser.parse_args()


conv_size = args.conv_size
order = args.order*0.01
lr=args.lr*0.00001
AlphaSet=args.alpha*0.1
num_sample=args.numsample
num_epoch =2000


resultpath = './results20200601/FGCN/muufl/order/'+"conv %g, order %g, lr %g, alpha %g, sam %g/"%(conv_size, order, lr, AlphaSet, num_sample)
if not os.path.isdir(resultpath):
    os.makedirs(resultpath)

Xpath='./DataSets/muufl.mat'
Xname='muufl'
Xshape=[1,326,220,64]
Xbands=64
XLpath='./DataSets/muuflL.mat'
XLname='muuflL'
XLshape=[1,326,220,2]
XLbands=2
Ypath='./DataSets/muufl_gt.mat'
Yname='muufl_gt'
Yshape=[1,326,220,11]
nclass=11

f = [1., 1./2, 1./3, 1./4]
filter_size_1 = [conv_size, conv_size, 64, 16]

gf_1 = gen_gf_list(filter_size_1, f, order)
gf_2 = gen_gf_list(filter_size_1, f, order)
gf_3 = gen_gf_list(filter_size_1, f, order)
filter_1 = filter_variable("filter_1",filter_size_1)

def weight_variable(shape,name=None):    
    initial = tf.random_uniform(shape,minval=0.01, maxval=0.02)
    return tf.Variable(initial,name)

def weight(name=None):
    initial = tf.random_uniform([1],minval=1, maxval=2)
    return tf.Variable(initial,name)

def bias_variable(shape,name=None):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial,name=None)

def conv2d(x, W, p=0):
    if p==1:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    else:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        
def atrousconv2d(x,W,rate):
    return tf.nn.atrous_conv2d(x, W,rate,padding='SAME')

def max_pool(x,W,p=0):
    if p==1:
        return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, W, W, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, W, W, 1], padding='VALID')        
                        
def max_pool_padding(x,W):
    return tf.nn.max_pool(x, ksize=[1, W, W, 1],
                        strides=[1, 1, 1, 1], padding='SAME')
                                               
def avg_pool_padding(x,W):
    return tf.nn.avg_pool(x, ksize=[1, W, W, 1],
                        strides=[1, 1, 1, 1], padding='SAME')                        
                        
def one_hot3d(Y):
    num_class = np.max(Y)
    row,col = Y.shape
    y = np.zeros((row,col,num_class),'uint8')
    for i in range(1,num_class+1):
        index = np.where(Y==i)
        y[index[0],index[1],i-1] = 1
    return y         

x = tf.placeholder(tf.float32, shape=Xshape)
xl = tf.placeholder(tf.float32, shape=XLshape)
y_ = tf.placeholder(tf.float32, shape=Yshape)

idx1 = tf.placeholder(tf.int32, shape=None)
idx2 = tf.placeholder(tf.int32, shape=None)
idx3 = tf.placeholder(tf.int32, shape=None)

W_spectral_conv1 = weight_variable([1,1, Xbands, 64],'W_spectral_conv1')
b_spectral_conv1 = bias_variable([Xbands],'b_spectral_conv1')
h_spectral_conv1 = tf.nn.relu(conv2d(x, W_spectral_conv1,1) + b_spectral_conv1)
h_spectral_conv1 = tf.nn.dropout(h_spectral_conv1,0.3)

W_spectral_conv2 = weight_variable([1,1, 64, 64],'W_spectral_conv2')
b_spectral_conv2 = bias_variable([64],'b_spectral_conv2')
h_spectral_conv2 = tf.nn.relu(conv2d(h_spectral_conv1, W_spectral_conv2,1) + b_spectral_conv2)
h_spectral_conv2 = tf.nn.dropout(h_spectral_conv2,0.3)

W_spectral_conv3 = weight_variable([1,1,64,64],'W_spectral_conv3')
b_spectral_conv3 = bias_variable([64],'b_spectral_conv3')
h_spectral_conv3 = tf.nn.relu(conv2d(h_spectral_conv2,W_spectral_conv3,1) + b_spectral_conv3)
h_spectral_conv3 = tf.nn.dropout(h_spectral_conv3,0.3)

h_spectral = tf.concat([h_spectral_conv1,h_spectral_conv2,h_spectral_conv3],3)


stride=(1, 1) 
hf_data_m, lf_data_m = firstOctConv_BN_AC(data=x, alpha=AlphaSet, num_filter_in=Xbands, num_filter_out=Xbands, kernel=( 1, 1), pad='valid')
hf_data_ml, lf_data_ml = firstOctConv_BN_AC(data=xl, alpha=AlphaSet, num_filter_in=XLbands, num_filter_out=Xbands, kernel=( 1, 1), pad='valid')
hf_data_merge1=tf.concat([hf_data_m, hf_data_ml],3)
lf_data_merge1=tf.concat([lf_data_m, lf_data_ml],3)
merge1, merge2 = octConv_BN_AC(hf_data=hf_data_merge1, lf_data=lf_data_merge1, alpha=AlphaSet, num_filter_in=Xbands*2,num_filter_out=Xbands*2, kernel=(3, 3), pad='same',stride=stride)
lidar_dr_conv1 = lastOctConv_BN_AC(hf_data=merge1, lf_data=merge2, alpha=AlphaSet, num_filter_in=Xbands*2,num_filter_out=64, kernel=(3, 3), pad='same',stride=stride)


h_dr_conv1  =   lidar_dr_conv1
b_spatial_conv1_1 = bias_variable([16],'b_spatial_conv1_1')
b_spatial_conv1_2 = bias_variable([16],'b_spatial_conv1_2')
b_spatial_conv1_3 = bias_variable([16],'b_spatial_conv1_3')
b_spatial_conv1_4 = bias_variable([16],'b_spatial_conv1_4')
h_spatial_conv1_1 = tf.nn.relu(atrousconv2d(h_dr_conv1, filter_1*gf_1[0],2) + b_spatial_conv1_1)
h_spatial_conv1_2 = tf.nn.relu(atrousconv2d(h_dr_conv1, filter_1*gf_1[1],2) + b_spatial_conv1_2)
h_spatial_conv1_3 = tf.nn.relu(atrousconv2d(h_dr_conv1, filter_1*gf_1[2],2) + b_spatial_conv1_3)
h_spatial_conv1_4 = tf.nn.relu(atrousconv2d(h_dr_conv1, filter_1*gf_1[3],2) + b_spatial_conv1_4)
h_spatial_conv1 = tf.concat([h_spatial_conv1_1,h_spatial_conv1_2,h_spatial_conv1_3,h_spatial_conv1_4],3)
h_spatial_conv1 = tf.nn.relu(h_spatial_conv1)
h_spatial_conv1 = tf.nn.dropout(h_spatial_conv1,0.3)
h_spatial_pool1 = max_pool_padding(h_spatial_conv1,2)

b_spatial_conv2_1 = bias_variable([16],'b_spatial_conv2_1')
b_spatial_conv2_2 = bias_variable([16],'b_spatial_conv2_2')
b_spatial_conv2_3 = bias_variable([16],'b_spatial_conv2_3')
b_spatial_conv2_4 = bias_variable([16],'b_spatial_conv2_4')
h_spatial_conv2_1 = tf.nn.relu(atrousconv2d(h_spatial_pool1,filter_1*gf_2[0],2) + b_spatial_conv2_1)
h_spatial_conv2_2 = tf.nn.relu(atrousconv2d(h_spatial_pool1,filter_1*gf_2[1],2) + b_spatial_conv2_2)
h_spatial_conv2_3 = tf.nn.relu(atrousconv2d(h_spatial_pool1,filter_1*gf_2[2],2) + b_spatial_conv2_3)
h_spatial_conv2_4 = tf.nn.relu(atrousconv2d(h_spatial_pool1,filter_1*gf_2[3],2) + b_spatial_conv2_4)
h_spatial_conv2 = tf.concat([h_spatial_conv2_1,h_spatial_conv2_2,h_spatial_conv2_3,h_spatial_conv2_4],3)
h_spatial_conv2 = tf.nn.relu(h_spatial_conv2)
h_spatial_conv2 = tf.nn.dropout(h_spatial_conv2,0.3)
h_spatial_pool2 = max_pool_padding(h_spatial_conv2,2)


b_spatial_conv3_1 = bias_variable([16],'b_spatial_conv3_1')
b_spatial_conv3_2 = bias_variable([16],'b_spatial_conv3_2')
b_spatial_conv3_3 = bias_variable([16],'b_spatial_conv3_3')
b_spatial_conv3_4 = bias_variable([16],'b_spatial_conv3_4')
h_spatial_conv3_1 = tf.nn.relu(atrousconv2d(h_spatial_pool2,filter_1*gf_3[0],2) + b_spatial_conv3_1)
h_spatial_conv3_2 = tf.nn.relu(atrousconv2d(h_spatial_pool2,filter_1*gf_3[1],2) + b_spatial_conv3_2)
h_spatial_conv3_3 = tf.nn.relu(atrousconv2d(h_spatial_pool2,filter_1*gf_3[2],2) + b_spatial_conv3_3)
h_spatial_conv3_4 = tf.nn.relu(atrousconv2d(h_spatial_pool2,filter_1*gf_3[3],2) + b_spatial_conv3_4)
h_spatial_conv3 = tf.concat([h_spatial_conv3_1,h_spatial_conv3_2,h_spatial_conv3_3,h_spatial_conv3_4],3)
h_spatial_conv3 = tf.nn.relu(h_spatial_conv3)
h_spatial_conv3 = tf.nn.dropout(h_spatial_conv3,0.3)
h_spatial_pool3 = max_pool_padding(h_spatial_conv3,2)

h_spatial = tf.concat([h_spatial_pool1, h_spatial_pool2, h_spatial_pool3],3)# h_spatial_pool3

W_spatial = weight('W_spatial')
W_spectral = weight('W_spectral')

h_SS = W_spatial*h_spatial + W_spectral*h_spectral 

W_conv5 = weight_variable([1,1, 192, 64],'W_conv5')
b_conv5 = bias_variable([64],'b_conv5')
y_conv = tf.nn.relu(conv2d(h_SS, W_conv5,1) + b_conv5)

W_conv6 = weight_variable([1,1, 64, nclass],'W_conv6')
b_conv6 = bias_variable([nclass],'b_conv6')
y_conv = tf.nn.relu(conv2d(y_conv, W_conv6,1) + b_conv6)


y_prob = tf.nn.softmax(y_conv)
y_label = tf.argmax(y_prob,-1)

indices = tf.stack([idx3,idx1,idx2], axis=1)

y_conv_mask = tf.gather_nd(y_conv, indices)
y_mask = tf.gather_nd(y_, indices)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv_mask, labels=y_mask))
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv_mask,-1), tf.argmax(y_mask,-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

X = sio.loadmat(Xpath)[Xname].astype('float32')
XL = sio.loadmat(XLpath)[XLname].astype('float32')
Y = sio.loadmat(Ypath)[Yname]

row,col,n_band = X.shape
Lrow,Lcol,Ln_band = XL.shape
X = np.reshape(featureNormalize(np.reshape(X,-1),2),(row,col,n_band))
XL = np.reshape(featureNormalize(np.reshape(XL,-1),2),(Lrow,Lcol,Ln_band))

num_class = np.max(Y)
Y_train = np.zeros(Y.shape).astype('int')
n_sample_train = 0
n_sample_test = 0

FCN_joint = np.zeros((num_class+4))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666) config=tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session()   
init = tf.global_variables_initializer()  
sess.run(init)

for i in range(1,num_class+1):
    index = np.where(Y==i)
    n_sample = len(index[0])
    array = np.random.permutation(n_sample)
    n_per = num_sample
    if i==1:
        array1_train = index[0][array[:n_per]]
        array2_train = index[1][array[:n_per]]
        array1_test = index[0][array[n_per:]]
        array2_test = index[1][array[n_per:]]
    else:
        array1_train = np.concatenate((array1_train,index[0][array[:n_per]]))
        array2_train = np.concatenate((array2_train,index[1][array[:n_per]]))
        array1_test = np.concatenate((array1_test,index[0][array[n_per:]]))
        array2_test = np.concatenate((array2_test,index[1][array[n_per:]]))
    Y_train[index[0][array[:n_per]],index[1][array[:n_per]]] = i
    n_sample_train += n_per
    n_sample_test += n_sample-n_per
    
array3 = np.zeros(array1_train.shape)

y_train = one_hot3d(Y_train)
y_train = np.reshape(y_train,(1,row,col,num_class))
X_train = np.reshape(X,(1,row,col,n_band))
XL_train = np.reshape(XL,(1,Lrow,Lcol,Ln_band))

mask_train = np.zeros(y_train.shape)
for i in range(num_class):
    mask_train[:,:,:,i] = np.sum(y_train,-1)
    
time1 = time.time()

histloss = np.zeros((num_epoch,2))

for i in range(num_epoch):

    train_accuracy = 0
    train_loss = 0
    
    sess.run(train_step,feed_dict={x:X_train,xl:XL_train,y_:y_train,idx1:array1_train,idx2:array2_train,idx3:array3})
    
    if (i+1)%100==0:
        train_accuracy,train_loss = sess.run([accuracy,cross_entropy],feed_dict={x:X_train,xl:XL_train,y_:y_train,idx1:array1_train,idx2:array2_train,idx3:array3})
       
        histloss[i,:] = train_accuracy,train_loss
    
        print("epoch %d, train_accuracy %g, train_loss %g"%(i+1, histloss[i,0],histloss[i,1]))

time2 = time.time()            
label = np.squeeze(sess.run(y_label,feed_dict={x:X_train,xl:XL_train,y_: y_train,idx1:array1_train,idx2:array2_train,idx3:array3}))
prob = np.squeeze(sess.run(y_prob,feed_dict={x:X_train,xl:XL_train,y_: y_train,idx1:array1_train,idx2:array2_train,idx3:array3}))    

y_test = Y[array1_test,array2_test]-1
y_pred = label[array1_test,array2_test]

OA,kappa,ProducerA = CalAccuracy(y_pred,y_test)
print("OA %g, kappa %g"%(OA, kappa))
FCN_joint[:num_class] = ProducerA
FCN_joint[-4] = OA
FCN_joint[-3] = kappa

time3 = time.time()    
FCN_joint[-2] = time2 - time1
FCN_joint[-1] = time3 - time2
print("Running time: %g"%(FCN_joint[-1]))
# sio.savemat(resultpath+'prob.mat', {'FCN_joint': prob})
# sio.savemat(resultpath+'label.mat', {'FCN_joint': label})
# sio.savemat(resultpath+repr(int(OA*10000))+'result.mat', {'FCN_joint': FCN_joint})