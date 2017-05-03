import os
import h5py
import numpy as np
import pandas as pd 
np.random.seed(148)

import sklearn.metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import backend as K
from keras.optimizers import SGD, RMSprop

from scipy import ndimage


def preprocess_input_res(x):
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def plot_confusion_matrix(cm):
    
    plt.imshow(cm, interpolation='nearest', cmap='hot')
    plt.colorbar()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusionmatrix2.svg')

# path to the model weights file.
img_width, img_height = 224, 224 
validation_data_dir = 'crop_v'


### Parameters
batch_size = 32
num_classes = 200
total_test = 5794
steps = int(total_test/batch_size)

DIR = './'
test_CSV = pd.read_csv(DIR + 'test.csv')
true_labels = test_CSV['label'].values[0:steps * batch_size]

testgen = ImageDataGenerator(preprocessing_function=preprocess_input_res)

generator_test = testgen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

base_model = ResNet50(weights='imagenet', input_shape=(img_width, img_height, 3), pooling='avg', include_top=False)

# Top classifier
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x)
predictions = Dense(num_classes, activation='softmax')(x)



# Combined model w/ classifier
model = Model(input=base_model.input, output=predictions)
model.summary()
model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('p2resnet_total.h5')

predictions = model.predict_generator(generator_test, steps, verbose=1)
pred_labels = np.array(map(np.argmax, predictions)) + 1

acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
cm = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=[i for i in range(1, 201)])
print "Accuracy: " + str(acc)

d = pred_labels - true_labels
print 1 - np.count_nonzero(d) / float(len(true_labels))
np.savetxt("matches.csv", d, delimiter=",")

plot_confusion_matrix(cm)