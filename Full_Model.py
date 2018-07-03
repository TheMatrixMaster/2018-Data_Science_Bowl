import os
#import cv2
import sys
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

style.use('ggplot')

Train_dir = 'C:/Users/Stephen Lu/Desktop/2018 Data_Science_Bowl/stage1_train/'
Test_dir = 'C:/Users/Stephen Lu/Desktop/2018 Data_Science_Bowl/stage1_test/'

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

Train_IDs = next(os.walk(Train_dir))[1]
Test_IDs = next(os.walk(Test_dir))[1]

#Get and resize train images and masks
#Train_Data
X_train = np.zeros((len(Train_IDs), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#Train_Labels
Y_train = np.zeros((len(Train_IDs), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
'''
print('Getting and resizing train images and merging independant masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(Train_IDs), total=len(Train_IDs)):
	img = cv2.imread(Train_dir + id_ + '/images/' + id_ + '.png', 1)
	img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
	X_train[n] = img
	mask = np.zeros((IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)
	for mask_photo in next(os.walk(Train_dir + id_ + '/masks/'))[2]:
		mask_ = cv2.imread(Train_dir + id_ + '/masks/' + mask_photo, 0)
		mask_ = np.expand_dims(cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC), axis=-1) 
		#Returns array with highest color value. So black/black stays black, black/white becomes white and white/white stays white.
		#Merges masks
		mask = np.maximum(mask, mask_)
	Y_train[n] = mask		
'''
#Get and resize test images
X_test = np.zeros((len(Test_IDs), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print()
print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(Test_IDs), total=len(Test_IDs)):
	path = Test_dir + id_
	img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
	sizes_test.append([img.shape[0], img.shape[1]])
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	X_test[n] = img

#np.save('train_data.npy', X_train)
#np.save('train_labels.npy', Y_train)
#np.save('test_data.npy', X_test)

print()
print('DONE!')	

X = np.load('train_data.npy')
y = np.load('train_labels.npy')
X_test = np.load('test_data.npy')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

'''
# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X, y, validation_split=0.1, batch_size=8, epochs=30, 
                    callbacks=[earlystopper, checkpointer])
'''

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X[:int(X.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X[int(X.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))