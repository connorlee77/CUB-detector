import numpy as np 
import pandas as pd
import os
import cv2
from priors import getPriors

# x1, y1, x2, y2
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def random_color():
    return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)

loc = np.load('loc.npy')
conf = np.load('conf.npy')
priors = getPriors()

DIR = './'
TEST_DIR = 'validation/'
data = pd.read_csv(DIR + 'test_multibox.csv')

test_data = np.load('test.npy')[0:5792]

for i, row in data.iterrows():
	if i == 5792:
		break
	row_conf = conf[i]
	agmax, cmax = np.argmax(row_conf[0]), np.max(row_conf[0])

	row_loc = loc[i]
	residuals = row_loc[:,agmax]
	prior = priors[:,agmax]

	bbox = prior + residuals
	x1, x2, y1, y2, C, R = row['x1'], row['x2'], row['y1'], row['y2'], row['C'], row['R']
	path = row['path']
	C, R = 299, 299
	bbox_scaled = map(int, [bbox[0] * C, bbox[1]*C, bbox[2]*R, bbox[3]*R])
	true_box = map(int, [x1*C, x2*C, y1*R, y2*R])

	iou = bb_intersection_over_union(true_box, bbox_scaled)

	# directories = os.path.dirname(path)
	# src = TEST_DIR + path
	print prior*299
	prior = map(int, prior*299)
	
	pic = test_data[i]
	cv2.rectangle(pic, (true_box[0], true_box[2]), (true_box[1], true_box[3]), (255, 255, 255))
	cv2.rectangle(pic, (bbox_scaled[0], bbox_scaled[2]), (bbox_scaled[1], bbox_scaled[3]), (255, 255, 0))
	cv2.rectangle(pic, (prior[0], prior[2]), (prior[1], prior[3]), (0, 0, 0))
	cv2.imshow('image',pic)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#print iou


