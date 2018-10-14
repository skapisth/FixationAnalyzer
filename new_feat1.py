from keras.preprocessing import image
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import csv
from keras.applications.resnet50 import ResNet50
import time
import tensorflow as tf


from keras import backend as K

PATCH_SIZE = 9
BATCH_SIZE = 50
LAYER = 50


def get_activations(model, layer_idx, X_batch):
    """
    :param model: CNN model
    :param layer_idx: layer index
    :param X_batch: batch of images
    :return: features extracted
    """
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output])
    activations = get_activations([X_batch,0])
    return activations


def read_file(file):
    """
    :param file: csv input
    :return: each column as data frame
    """
    df0 = file
    df1 = df0['subject']
    df2 = df0['Subject-renamed']
    df3 = df0['Stimulus']
    df4 = df0['Stim-renamed']
    df5 = df0['exp']
    df6 = df0['inst']
    df7 = df0['PositionX']
    df8 = df0['PositionY']
    X = df7
    Y = df8
    return df1,df2,df3,df4,df5,df6,df7,df8,X,Y

def read_images(image_name,i):
    """
    :param image_name:
    :param i: integer
    :return: normalized image
    """
    print ("IMAGE: " + str(image_name) + ".jpg")
    image = os.path.join('./' + str(image_name) + '.jpg')
    im = plt.imread(image).astype(np.float32)
    im = im/np.amax(im)
    return im

def extract_im_patch(im,x,y):
    """
    :param im: image
    :param x: x-coordinate of fixation
    :param y: y-coordinate of fixation
    :return: nxn image patch
    """

    temp_matrix = PATCH_SIZE/2
    left_x   = x[i] - temp_matrix
    right_x  = x[i] + temp_matrix
    top_y    = y[i] - temp_matrix
    bottom_y = y[i] + temp_matrix
    im_patch = im[int(top_y):int(bottom_y),int(left_x):int(right_x)]
    return im_patch


def process_im_patch(im_patch):
    """
    :param im_patch: nxn image patch
    :return: processed image patch
    """
    img = np.expand_dims(im_patch,axis=0)
    img = preprocess_input(img)
    return img

def output_labels(i,df1,df2,df3,df4,df5,df6):
    """
    :param i: integer
    :param df1: dataframe with subject names
    :param df2: dataframe with subjects renamed to integers as 1,2,....
    :param df3: dataframe with stimuli names
    :param df4: dataframe with stimuli names renamed to 1,2,...
    :param df5: dataframe specifying whether expert or not with binary labels i.e., 0 and 1
    :param df6: dataframe specifying whether instructed or not with binary labels i.e., 0 and 1
    :return: list of all the input dataframes defined and given
    """
    sub.append(df1[i])
    sub_rename.append(df2[i])
    stim.append(df3[i])
    stim_rename.append(df4[i])
    exp.append(df5[i])
    inst.append(df6[i])
    return sub,sub_rename,stim,stim_rename,exp,inst

def save_output_labels(sub,sub_rename,stim,stim_rename,exp,inst):
    """
    :param sub: list of subject names
    :param sub_rename: list of re-named subjects
    :param stim: list of stimuli names
    :param stim_rename: list of re-named stimuli
    :param exp: Expert label - 1; Novice label - 0
    :param inst: Instructed label - 1, Not-instructed - 0
    :return:  All the labels concatenated into one dataframe
    """
    df10 = pd.DataFrame(sub)
    df11 = pd.DataFrame(sub_rename)
    df12 = pd.DataFrame(stim)
    df13 = pd.DataFrame(stim_rename)
    df14 = pd.DataFrame(exp)
    df15 = pd.DataFrame(inst)
    df17 = pd.concat([df10,df11,df12,df13,df14,df15],axis=1)
    return df17



file = "./clean_data.csv"
df0 = pd.read_csv(file)
df1,df2,df3,df4,df5,df6,df7,df8,X,Y  = read_file(df0)



model = ResNet50(weights='imagenet',include_top=False)
features = []
im_patches = []
exp = []
inst = []
sub = []
sub_rename = []
stim = []
stim_rename = []
with tf.device("/gpu:0"):
    for i in range(len(df4)):
        image_name = df4[i]
        x = X
        y = Y
        #import pdb;pdb.set_trace()
        if np.any(i>0) and np.any(image_name==df4[i-1]):
            im = im
        else:
            im =  read_images(image_name,i)

        if np.any(x[i] < self.patch_size/2):
            continue
        if np.any(x[i] > width-self.patch_size/2):
            continue
        if np.any(y[i] < self.patch_size/2):
            continue
        if np.any(y[i] > height-self.patch_size/2):
            continue

        sub,sub_rename,stim,stim_rename,exp,inst = output_labels(i,df1,df2,df3,df4,df5,df6)
        im_patch = extract_im_patch(im,x,y)
        processed_im_patch = process_im_patch(im_patch)
        im_patches.append(processed_im_patch)

all_im_patches = np.vstack(im_patches)


for k in range(len(all_im_patches)):
    startTime = time.time()
    print ("loop_num: " + str(k))
    print ("num_patches: " +str(len(all_im_patches)))
    #num = 70
    if np.any(k>0):
        start = BATCH_SIZE*k
        stop = BATCH_SIZE*(k+1)
        imgs = all_im_patches[start:stop,:,:,:]

        if np.any(len(imgs)==BATCH_SIZE):
            activations = get_activations(model, LAYER, imgs)
        elif np.any(len(imgs)<BATCH_SIZE and len(imgs)!=0):
            activations = get_activations(model, LAYER, imgs)
        elif np.any(len(imgs)==0):
            print ("All Done!")
            break
    else:
        start = k
        stop = BATCH_SIZE
        imgs = all_im_patches[start:stop,:,:,:]
        activations = get_activations(model, LAYER, imgs)
    feat = np.reshape(activations[0],(activations[0].shape[0],activations[0].shape[1]*activations[0].shape[2]*activations[0].shape[3]))
    features.append(feat)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        tensor_object = tf.convert_to_tensor(feat)
        sess.run(tensor_object)
        stopTime = time.time()
        print ("Time: " + str(stopTime - startTime))


all_feat = np.concatenate(features,axis=0)
np.savetxt('features_high_50.csv',all_feat,delimiter=',')
df17 = save_output_labels(sub,sub_rename,stim,stim_rename,exp,inst)
df17.to_csv('features_high_50_labels.csv')
