import os
import shutil

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import skimage.io

DIR = './'
TRAIN_DIR = 'train/'
TEST_DIR = 'validation/'
IMAGE_DIR = 'images/'

def toCSV():

	images = pd.read_csv(DIR + 'images.txt', delim_whitespace=True, header=None, names=['id', 'path'])
	datasetType = pd.read_csv(DIR + 'train_test_split.txt', delim_whitespace=True, header=None, names=['id', 'set'])
	labels = pd.read_csv(DIR + 'image_class_labels.txt', delim_whitespace=True, header=None, names=['id', 'label'])
	boxes = pd.read_csv(DIR + 'bounding_boxes.txt', delim_whitespace=True, header=None, names=['id', 'x', 'y', 'width', 'height'])

	# id (int) | training or testing (int) | class label (int) | x | y | width | height
	data = pd.concat([images, datasetType['set'], labels['label'], boxes['x'], boxes['y'], boxes['width'], boxes['height']], axis=1)

	train = data[data['set'] == 1].reset_index(drop=True)
	test = data[data['set'] == 0].reset_index(drop=True)

	for index, row in train.iterrows():
		path = row['path'] 

		directories = os.path.dirname(path)

		src = IMAGE_DIR + path
		dst = TRAIN_DIR + path

		try:
			os.makedirs(TRAIN_DIR + directories)
			shutil.copy(src, dst)
		except OSError:
			shutil.copy(src, dst)

	for index, row in test.iterrows():
		path = row['path'] 

		directories = os.path.dirname(path)

		src = IMAGE_DIR + path
		dst = TEST_DIR + path

		try:
			os.makedirs(TEST_DIR + directories)
			shutil.copy(src, dst)
		except OSError:
			shutil.copy(src, dst)

	train.to_csv('train.csv', index=False)
	test.to_csv('test.csv', index=False)

def normalizeBoundingBox(data, name, data_dir):
	
	x1d = []
	x2d = []
	y1d = []
	y2d = []
	C = []
	R = []
	for index, row in data.iterrows():
		path, x, y, width, height = row['path'], int(row['x']), int(row['y']), int(row['width']), int(row['height'])

		src = data_dir + path

		pic = skimage.io.imread(src)
		shape = np.float32(pic.shape)
		r = shape[0]
		c = shape[1]

		x1 = x 
		x2 = x + width
		
		y1 = y 
		y2 = y + height

		x1n = x1 / c 
		assert x2 / c < 1.1
		x2n = min(1.0, x2 / c)
		
		y1n = y1 / r
		assert y2 / r < 1.1
		y2n = min(1.0, y2 / r)
		
		x1d.append(x1n)
		x2d.append(x2n)
		y1d.append(y1n)
		y2d.append(y2n)
		C.append(c)
		R.append(r)

	x1 = pd.Series(x1d, index=data.index, name='x1')
	x2 = pd.Series(x2d, index=data.index, name='x2')
	y1 = pd.Series(y1d, index=data.index, name='y1')
	y2 = pd.Series(y2d, index=data.index, name='y2')
	C = pd.Series(C, index=data.index, name='C')
	R = pd.Series(R, index=data.index, name='R')

	newFrame = pd.concat([data['id'], data['path'], x1, x2, y1, y2, C, R], axis=1)
	newFrame.to_csv(name, index=False)

def boundingbox():
	train_CSV = pd.read_csv(DIR + 'train.csv')
	test_CSV = pd.read_csv(DIR + 'test.csv')


	normalizeBoundingBox(train_CSV, 'train_multibox.csv', TRAIN_DIR)
	normalizeBoundingBox(test_CSV, 'test_multibox.csv', TEST_DIR)

boundingbox()