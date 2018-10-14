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

#model = Sequential()
#model.layers.pop()
#model.layers.pop()
#model.add(ZeroPadding2D((1,1),input_shape=(8,8,3)))
#model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(BatchNormalization(mode=0, axis=1))
#model.add(ZeroPadding2D((1,1)))
#model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(BatchN

from keras import backend as K
def get_activations(model, layer_idx, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
    activations = get_activations([X_batch,0])
    return activations
file = "./trial_data.csv"
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
im_patches = []
list_ix = list(range(len(df4)))
list_ix_x = list(range(len(df7)))
list_ix_y = list(range(len(df8)))
idxs = pd.Series(df4,index=list_ix)
idx_X = pd.Series(df7,index=list_ix_x)
idx_Y = pd.Series(df8,index=list_ix_y)
for i in xrange(10):
    image_name = 18 #df4[i]
    #ix = list(idxs[idxs==i+1].index)
    x = X#[ix]
    y = Y#[ix]
    if np.any(i>0) and np.any(image_name==df4[i-1]):
        im = im
    else:
        image = os.path.join('./' + str(image_name) + '.jpg')
        im = plt.imread(image).astype(np.float32)
        im = im/np.amax(im)
    cut_w = 0.1*im.shape[0]
    cut_h = 0.1*im.shape[1]
    matrix_size =9
    temp_matrix = matrix_size/2
    model = ResNet50(weights='imagenet',include_top=False)
    import pdb;pdb.set_trace()
    blue = im[:,:,0]
    green = im[:,:,1]
    red = im[:,:,2]
    #import pdb;pdb.set_trace()
    #for k in xrange(len(x)):
    lim_left_x = 0 + (cut_w/2)
    lim_right_x = im.shape[0]- (cut_w/2)
    lim_top_y = 0 + (cut_h/2)
    lim_bot_y = im.shape[1] + (cut_h/2)
    if np.any(x[i] < lim_left_x):
            #import pdb;pdb.set_trace()                                                                                                                                            
        continue
    if np.any(x[i] > lim_right_x):                                                                                        
            #import pdb;pdb.set_trace()                                                                                                                                            
        continue
    if np.any(y[i] < lim_top_y):
        #import pdb;pdb.set_trace()                                                                                                                                            
        continue
    if np.any(y[i] > lim_bot_y):
        #import pdb;pdb.set_trace()                                                                                                                                            
        continue
    #idx_of_x = list(idx_X[idx_X==x[k]].index)
    #idx_of_y = list(idx_Y[idx_Y==y[k]].index)
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
    b_patch = blue[top_y:bottom_y,left_x:right_x]
    g_patch = green[top_y:bottom_y,left_x:right_x]
    r_patch = red[top_y:bottom_y,left_x:right_x]
    act_patch = cv2.merge((b_patch,g_patch,r_patch))
    img = act_patch#.astype(np.float32)
    #import pdb;pdb.set_trace()
    #img = np.resize(img,(197,197,3))
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    import pdb;pdb.set_trace()
    #feat1 = model.predict(img)
    im_patches.append(img)
    #import pdb;pdb.set_trace()
        #get_feature = theano.function([model.layers[0].input],model.layers[15].get_output(train=False),allow_input_downcast=False)
        #feat = get_feature(img)
all_im_patches = np.vstack(im_patches)
for i in xrange(len(all_im_patches)):
    import pdb;pdb.set_trace()
activations = get_activations(model,49, img)
feat = np.reshape(activations[0],(activations[0].shape[0],activations[0].shape[1]*activations[0].shape[2]*activations[0].shape[3]))
features.append(feat)

import pdb;pdb.set_trace()
df10 = pd.DataFrame(sub)
df11 = pd.DataFrame(sub_rename)
df12 = pd.DataFrame(stim)
df13 = pd.DataFrame(stim_rename)
df14 = pd.DataFrame(exp)
df15 = pd.DataFrame(inst)
df16 = pd.DataFrame(features)


df17 = pd.concat([df10,df11,df12,df13,df14,df15,df16])
df17.to_csv('features.csv')
