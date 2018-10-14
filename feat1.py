from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import pandas as pd
import csv
import cv2
import theano
from keras.applications.resnet50 import ResNet50
import time
import tensorflow as tf


from keras import backend as K
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output])
    activations = get_activations([X_batch,0])
    return activations
file = "./clean_data.csv"
df0 = pd.read_csv(file)
df1 = df0['subject']
df2 = df0['Subject-renamed']
df3 = df0['Stimulus']
df4 = df0['Stim-renamed']
df5 = df0['exp']
df6 = df0['inst']
df7 = df0['PositionX']
df8 = df0['PositionY']
X = df7#.astype(np.uint8)                                                     
Y = df8#.astype(np.uint8)                                                     
                                                   

features = []
exp = []
inst = []
sub = []
sub_rename = []
stim = []
stim_rename = []
list_ix = list(range(len(df4)))
list_ix_x = list(range(len(df7)))
list_ix_y = list(range(len(df8)))
idxs = pd.Series(df4,index=list_ix)
idx_X = pd.Series(df7,index=list_ix_x)
idx_Y = pd.Series(df8,index=list_ix_y)
im_patch = []
model = ResNet50(weights='imagenet',include_top=False)     
with tf.device("/gpu:0"):
    for i in range(len(df4)):
        
        image_name = df4[i]
        #ix = list(idxs[idxs==i+1].index)
        x = X#[ix]
        y = Y#[ix]
        if np.any(i>0) and np.any(image_name==df4[i-1]):
            im = im
        else:
            print ("IMAGE: " + str(image_name) + ".jpg")
            image = os.path.join('./' + str(image_name) + '.jpg')
            im = plt.imread(image).astype(np.float32)
            im = im/np.amax(im)
        width = im.shape[1]
	height = im.shape[0]
	matrix_size = 9
	temp_matrix = matrix_size/2
	if np.any(x[i] < temp_matrix):
	    continue
	if np.any(x[i] > width-temp_matrix):
            continue
        if np.any(y[i] < temp_matrix):
	    continue
        exp.append(df5[i])
        inst.append(df6[i])
        sub.append(df1[i])
        sub_rename.append(df2[i])
        stim.append(df3[i])
        stim_rename.append(df4[i])
        left_x   = x[i] - temp_matrix
        right_x  = x[i] + temp_matrix
        top_y    = y[i] - temp_matrix
        bottom_y = y[i] + temp_matrix
        
        act_patch = im[int(top_y):int(bottom_y),int(left_x):int(right_x)]
        img = act_patch
        img = np.expand_dims(img,axis=0)
        #import pdb;pdb.set_trace()
        #if np.any(i==349):
            #import pdb;pdb.set_trace()
        img = preprocess_input(img)
        im_patch.append(img)
        #import pdb;pdb.set_trace()
            
all_im_patches = np.vstack(im_patch)
for k in range(len(all_im_patches)):
    startTime = time.time()
    print ("loop_num: " + str(k))
    print ("num_patches: " +str(len(all_im_patches)))
    num = 70
    if np.any(k>0):
        start = num*k
        stop = num*(k+1)#+1
        imgs = all_im_patches[start:stop,:,:,:] 
        #import pdb;pdb.set_trace()
        if np.any(len(imgs)==num):
            activations = get_activations(model, 6, imgs)
        elif np.any(len(imgs)<num and len(imgs)!=0):
            #import pdb;pdb.set_trace()
            activations = get_activations(model, 6, imgs)
        elif np.any(len(imgs)==0):
            print ("nothing") 
            break
    else:
        start = k
        stop = num
        imgs = all_im_patches[start:stop,:,:,:]
        activations = get_activations(model, 6, imgs)
    #import pdb;pdb.set_trace()    
    feat = np.reshape(activations[0],(activations[0].shape[0],activations[0].shape[1]*activations[0].shape[2]*activations[0].shape[3]))
    features.append(feat)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # Run the op.
        #print ("BEFORE SESSION RUN")
        tensor_object = tf.convert_to_tensor(feat)
        sess.run(tensor_object)
        #print ("AFTER SESSION RUN")
        stopTime = time.time()
        print ("Time: " + str(stopTime - startTime))
            
		
#import pdb;pdb.set_trace()
df10 = pd.DataFrame(sub)
df11 = pd.DataFrame(sub_rename)
df12 = pd.DataFrame(stim)
df13 = pd.DataFrame(stim_rename)
df14 = pd.DataFrame(exp)
df15 = pd.DataFrame(inst)
#df16 = pd.DataFrame(features)


df17 = pd.concat([df10,df11,df12,df13,df14,df15],axis=1)
all_feat = np.concatenate(features,axis=0)
np.savetxt('features_low_6_new.csv',all_feat,delimiter=',')
df17.to_csv('features_low_6_labels_new.csv')
