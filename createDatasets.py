import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import skimage.io
from skimage import transform as tf 
import cv2 

DIR = './'
TRAIN_DIR = 'train/'
TEST_DIR = 'validation/'


train_CSV = pd.read_csv(DIR + 'train_multibox.csv')
test_CSV = pd.read_csv(DIR + 'test_multibox.csv')

def createSet(data, directory, filename, width, height):
	x = np.empty((len(data), width, height, 3))

	for index, row in data.iterrows():
		path = row['path']

		directories = os.path.dirname(path)
		src = directory + path
		pic = cv2.imread(src)
		pic = cv2.resize(pic, (height, width))
		assert pic.shape == (299, 299, 3)
		x[index] = pic

	x = np.uint8(x)
	np.save(filename, x)

createSet(train_CSV, TRAIN_DIR, 'train.npy', 299, 299)
createSet(test_CSV, TEST_DIR, 'test.npy', 299, 299)
