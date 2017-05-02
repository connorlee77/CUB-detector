import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import h5py
import numpy as np
import pandas as pd
np.random.seed(148)

import keras
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, Lambda
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import backend as K
from keras.optimizers import SGD, RMSprop

from priors import getPriors


def fixLabels(test):
    a = np.zeros((len(test), 4, 1419))
    for i, row in enumerate(test):
        a[i] = np.repeat(np.array([row]).T, 1419, axis=1)
    return a

# File paths
DIR = './'
TRAIN = 'train.npy'
TEST = 'test.npy'

# Data gathering & preprocessing
x_test = np.float32(np.load(TEST))
test_CSV = pd.read_csv(DIR + 'test_multibox.csv')

y_test = np.float32(fixLabels(test_CSV.as_matrix(['x1', 'x2', 'y1', 'y2'])))
x_test = preprocess_input(x_test)

test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow(x_test, y_test, batch_size=32, shuffle=False)

priors = getPriors()
priors = K.variable(priors)

### Parameters
img_width, img_height = 299, 299 
batch_size = 32
test_size = len(x_test)


base_model = InceptionV3(weights='imagenet', input_shape=(img_width, img_height, 3), pooling=None, include_top=False)

# Top classifier
x = base_model.output
### Global Avg Pooling 

### Branch A
a_br1 = ZeroPadding2D(padding=0)(x)
a_br1 = Conv2D(filters=96, kernel_size=1, strides=1, activation='relu')(a_br1)
# Branch A, Layer 2
a_br1 = ZeroPadding2D(padding=1)(a_br1)
a_br1 = Conv2D(filters=96, kernel_size=3, strides=1, activation='relu')(a_br1)
# Branch A, Outputs
a_br1_loc = Conv2D(filters=44, kernel_size=1, name='a_br1_loc')(a_br1)
a_br1_conf = Conv2D(filters=11, kernel_size=1, activation='sigmoid', name='a_br1_conf')(a_br1)

a_out_loc = Reshape((4, 8*8*11))(a_br1_loc)
a_out_conf = Reshape((1, 8*8*11))(a_br1_conf)

### Branch B
b_br1 = ZeroPadding2D(padding=1)(x)
b_br1 = Conv2D(filters=96, kernel_size=3, strides=1, activation='relu')(b_br1)
# Branch B, Layer 2
b_br1 = ZeroPadding2D(padding=0)(b_br1)
b_br1 = Conv2D(filters=96, kernel_size=3, strides=1, activation='relu')(b_br1)
# Branch B, Outputs
b_br1_loc = Conv2D(filters=44, kernel_size=1, name='b_br1_loc')(b_br1)
b_br1_conf = Conv2D(filters=11, kernel_size=1, activation='sigmoid', name='b_br1_conf')(b_br1)

b_out_loc = Reshape((4, 6*6*11))(b_br1_loc)
b_out_conf = Reshape((1, 6*6*11))(b_br1_conf)

### Branch C
c_br1 = ZeroPadding2D(padding=2)(x)
c_br1 = Conv2D(filters=256, kernel_size=3, strides=3, activation='relu')(c_br1)
# Branch C, Layer 2
c_br1a = ZeroPadding2D(padding=1)(c_br1)
c_br1a = Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')(c_br1a)

c_br1b = ZeroPadding2D(padding=0)(c_br1)
c_br1b = Conv2D(filters=128, kernel_size=1, strides=1, activation='relu')(c_br1b)

c_br1c = ZeroPadding2D(padding=0)(c_br1)
c_br1c = Conv2D(filters=128, kernel_size=1, strides=1, activation='relu')(c_br1c)
# Branch C, Layer 3
c_br1b = ZeroPadding2D(padding=0)(c_br1)
c_br1b = Conv2D(filters=96, kernel_size=2, strides=1, activation='relu')(c_br1b)

c_br1c = ZeroPadding2D(padding=0)(c_br1)
c_br1c = Conv2D(filters=96, kernel_size=3, strides=1, activation='relu')(c_br1c)
# Branch C, Outputs
c_br1a_loc = Conv2D(filters=44, kernel_size=1, name='c_br1a_loc')(c_br1a)
c_br1a_conf = Conv2D(filters=11, kernel_size=1, activation='sigmoid', name='c_br1a_conf')(c_br1a)

c_br1b_loc = Conv2D(filters=44, kernel_size=1, name='c_br1b_loc')(c_br1b)
c_br1b_conf = Conv2D(filters=11, kernel_size=1, activation='sigmoid', name='c_br1b_conf')(c_br1b)

c_br1c_loc = Conv2D(filters=44, kernel_size=1, name='c_br1c_loc')(c_br1c)
c_br1c_conf = Conv2D(filters=11, kernel_size=1, activation='sigmoid', name='c_br1c_conf')(c_br1c)

c_a_out_loc = Reshape((4, 4*4*11))(c_br1a_loc)
c_a_out_conf = Reshape((1, 4*4*11))(c_br1a_conf)
c_b_out_loc = Reshape((4, 3*3*11))(c_br1b_loc)
c_b_out_conf = Reshape((1, 3*3*11))(c_br1b_conf)
c_c_out_loc = Reshape((4, 2*2*11))(c_br1c_loc)
c_c_out_conf = Reshape((1, 2*2*11))(c_br1c_conf)

loc_concat = concatenate([a_out_loc, b_out_loc, c_a_out_loc, c_b_out_loc, c_c_out_loc], name='loc')
conf_concat = concatenate([a_out_conf, b_out_conf, c_a_out_conf, c_b_out_conf, c_c_out_conf], name='conf')

alpha = 0.7
def F(y_true, y_pred):
    predicted_positions = y_pred + priors     
    F_conf = -K.log(conf_concat)    
    F_loc = K.sqrt(K.sum(K.square(predicted_positions - y_true), axis=1, keepdims=True)) / 2.0    
    F_loss = F_conf + alpha * F_loc    
    F_min = K.min(F_loss)    
    return F_min

# Combined model w/ classifier
model = Model(inputs=base_model.input, outputs=loc_concat)
model.load_weights('inception_top.h5')

model.summary()
model.compile(optimizer=SGD(lr=0.01, decay=0.99), loss=F)

intermediate_layer_model = Model(inputs=base_model.input, outputs=[loc_concat, conf_concat])
intermediate_output = intermediate_layer_model.predict_generator(generator=validation_generator, steps=test_size/batch_size, verbose=1)

loc = intermediate_output[0]
conf = intermediate_output[1]

np.save('loc.npy', loc)
np.save('conf.npy', conf)