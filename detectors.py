import numpy as np 
import pandas as pd
import os
import cv2
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from priors import getPriors

def union(a, b):
	wa = a[1] - a[0]
	ha = a[3] - a[2]

	wb = b[1] - b[0]
	hb = b[3] - b[2]

	return wa*ha + wb*hb - intersection(a, b)

# xmin xmax ymin ymax
def intersection(a, b):

	dx = min(a[1], b[1]) - max(a[0], b[0])
	dy = min(a[3], b[3]) - max(a[2], b[2])
	if dx >= 0 and dy >= 0:
		return dx*dy
	return 0

def iou(a, b):
	return intersection(a, b) / float(union(a, b))

def random_color():
    return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)

def computeIOU(threshold):
	loc = np.load('loc1x1.npy')
	conf = np.load('conf1x1.npy')
	priors = getPriors()

	DIR = './'
	TEST_DIR = 'validation/'
	data = pd.read_csv(DIR + 'test_multibox.csv')
	test_data = np.load('test.npy')[0:5792]

	y_true = np.zeros((1, 5792))
	prob = np.zeros((1, 5792))

	for i, row in data.iterrows():
		if i == 5792:
			break
		row_conf = conf[i]
		agmax, cmax = np.argmax(row_conf), np.max(row_conf)

		row_loc = loc[i]
		residuals = row_loc[:,agmax]
		prior = priors[:,agmax]

		x1, x2, y1, y2, C, R = row['x1'], row['x2'], row['y1'], row['y2'], row['C'], row['R']
		path = row['path']
		
		C, R = 299, 299
		bbox = prior + residuals
		bbox_scaled = map(int, [bbox[0]*C, bbox[1]*C, bbox[2]*R, bbox[3]*R])
		true_box = map(int, [x1*C, x2*C, y1*R, y2*R])

		bbox_scaled = np.clip(bbox_scaled, 0, max(R, C))
		
		iou_score = iou(true_box, bbox_scaled)
		if iou_score > threshold:
			y_true[:,i] = 1
		prob[:,i] = cmax

		
	return prob, y_true 


def visualize(threshold, confidence=0.45):
	loc = np.load('loc_test.npy')
	conf = np.load('conf_test.npy')
	priors = getPriors()

	DIR = './'
	TEST_DIR = 'validation/'
	data = pd.read_csv(DIR + 'test_multibox.csv')

	test_data = np.load('test.npy')[0:5792]
	count = 0
	for i, row in data.iterrows():
		if i == 5792:
			break
		row_conf = conf[i]
		agmax, cmax = np.argmax(row_conf), np.max(row_conf)

		row_loc = loc[i]
		residuals = row_loc[:,agmax]
		prior = priors[:,agmax]

		
		x1, x2, y1, y2, C, R = row['x1'], row['x2'], row['y1'], row['y2'], row['C'], row['R']
		path = row['path']
		
		
		bbox = prior + residuals
		bbox = np.clip(bbox, 0, 1)
		bbox_scaled = map(int, [bbox[0]*C, bbox[1]*C, bbox[2]*R, bbox[3]*R])
		true_box = map(int, [x1*C, x2*C, y1*R, y2*R])
		
		iou_score = iou(true_box, bbox_scaled)
		
		if iou_score > threshold and cmax > confidence:
		
			directories = os.path.dirname(path)
			src = TEST_DIR + path
			a = map(int, [prior[0]*C, prior[1]*C, prior[2]*R, prior[3]*R])
			pic = cv2.imread(src)
			overlay = pic.copy()
			# pic = test_data[i]
			#cv2.rectangle(pic, (true_box[0], true_box[2]), (true_box[1], true_box[3]), (255, 255, 255))
			cv2.rectangle(overlay, (bbox_scaled[0], bbox_scaled[2]), (bbox_scaled[1], bbox_scaled[3]), (0, 0, 0), thickness=3)
			opacity = cmax
			cv2.addWeighted(overlay, opacity, pic, 1 - opacity, 0, pic)
			#cv2.rectangle(pic, (a[0], a[2]), (a[1], a[3]), (0, 0, 0))
			cv2.imshow('image', pic)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			#cv2.imwrite('./pics/' + str(i) + '.jpg', pic)

# visualize(threshold=0.5)

# for t in [0.3, 0.5, 0.7]:
# 	p, y = computeIOU(t)
# 	p, r, _ = precision_recall_curve(y[0], p[0], pos_label=1)
# 	plt.plot(r, p, label=str(t))

# plt.ylabel('Precision')
# plt.xlabel('Recall')
# plt.legend(loc=1)
# plt.show()

def f1(threshold):
	loc = np.load('loc1x1.npy')
	conf = np.load('conf1x1.npy')
	priors = getPriors()

	DIR = './'
	TEST_DIR = 'validation/'
	data = pd.read_csv(DIR + 'test_multibox.csv')
	test_data = np.load('test.npy')[0:5792]

	y_pred = np.zeros(5792)
	labels = np.zeros(5792)

	for i, row in data.iterrows():
		if i == 5792:
			break
		row_conf = conf[i]
		agmax, cmax = np.argmax(row_conf), np.max(row_conf)

		row_loc = loc[i]
		residuals = row_loc[:,agmax]
		prior = priors[:,agmax]

		x1, x2, y1, y2, C, R = row['x1'], row['x2'], row['y1'], row['y2'], row['C'], row['R']
		path = row['path']
		
		true_class = int(path[:3])
		labels[i] = true_class

		C, R = 299, 299
		bbox = prior + residuals
		bbox_scaled = map(int, [bbox[0]*C, bbox[1]*C, bbox[2]*R, bbox[3]*R])
		true_box = map(int, [x1*C, x2*C, y1*R, y2*R])

		bbox_scaled = np.clip(bbox_scaled, 0, max(R, C))
		
		iou_score = iou(true_box, bbox_scaled)
		if iou_score > threshold:
			y_pred[i] = 1


		
	return labels, y_pred

# labels, y_pred = f1(0.6)

# birds = {} 

# i = 0
# while i < len(labels):
# 	label = int(labels[i])
# 	pred = y_pred[i]

# 	if label in birds:
# 		birds[label].append(pred)
# 	else:
# 		birds[label] = [pred]
# 	i += 1

# fscores = np.zeros(200)
# i = 0
# while i < 200:
# 	f = f1_score(len(birds[i+1])*[1], birds[i+1])
# 	fscores[i] = f
# 	i += 1

# Y = fscores
# X = range(1, 201)
# r = [x for (y,x) in sorted(zip(Y,X))]

# print r
# print sorted(Y)
# fig,ax = plt.subplots()
# ax.bar(X, sorted(Y), color='b')
# ax.set_xticklabels([])
# plt.xlabel('Bird Species')
# plt.ylabel('F1 Score')
# plt.show()



def crop(DATA_DIR, DST_DIR, locations, confidence, DATA_CSV):
	loc = np.load(locations)
	conf = np.load(confidence)
	priors = getPriors()

	data = pd.read_csv(DATA_CSV)

	for i, row in data.iterrows():
		if i == 5792:
			break
		row_conf = conf[i]
		agmax, cmax = np.argmax(row_conf), np.max(row_conf)

		row_loc = loc[i]
		residuals = row_loc[:,agmax]
		prior = priors[:,agmax]

		x1, x2, y1, y2, C, R = row['x1'], row['x2'], row['y1'], row['y2'], row['C'], row['R']
		path = row['path']
		
		bbox = prior + residuals
		bbox = np.clip(bbox, 0, 1)
		bbox_scaled = map(int, [bbox[0]*C, bbox[1]*C, bbox[2]*R, bbox[3]*R])
		
		directories = os.path.dirname(path)
		src = DATA_DIR + path
		dst = DST_DIR + path

		pic = cv2.imread(src)
		cropped = pic[bbox_scaled[2]:bbox_scaled[3], bbox_scaled[0]:bbox_scaled[1]]
		
		if cropped.shape[0] == 0 or cropped.shape[1] == 0 or cropped.shape[2] == 0:
			cropped = pic 

		try:
			os.makedirs(DST_DIR + directories)
			cv2.imwrite(dst, cropped)
		except OSError:
			cv2.imwrite(dst, cropped)

crop('validation/', 'crop_v/', 'loc_test.npy', 'conf_test.npy', 'test_multibox.csv')
crop('train/', 'crop_t/', 'loc_train.npy', 'conf_train.npy', 'train_multibox.csv')