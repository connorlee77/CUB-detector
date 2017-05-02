import numpy as np 

import cv2
from skimage import data


def random_color():
    return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)

def getPriors():
	IMAGE_WIDTH = 299
	IMAGE_HEIGHT = 299
	gridDim = np.array([8, 6, 4, 3, 2])
	numPriors = np.square(gridDim) * 11

	aspectRatioW = np.linspace(start=2, stop=2.5, num=11)
	aspectRatioH = np.linspace(start=2, stop=1.5, num=11)

	priors = {}

	for x in gridDim:
		block = np.zeros((x, x, 44))
		# chelsea = cv2.resize(data.chelsea(), (299, 299))
		deltaQ = 1 / float(x + 1)
		for r in range(x):
			for c in range(x):

				curr = block[r, c]

				rn1 = r * deltaQ
				cn1 = c * deltaQ

				d = 0
				while d < 11:
					dw = aspectRatioW[d] * deltaQ
					dh = aspectRatioH[d] * deltaQ

					rn2 = rn1 + dh 
					cn2 = cn1 + dw 

					curr[4*d:4*d+4] = np.array([rn1, rn2, cn1, cn2])
					# print curr[4*d:4*d+4]
					# cv2.rectangle(chelsea, (int(rn1*299), int(cn1*299)), (int(rn2*299), int(cn2*299)), random_color())
					d += 1
		priors[x] = block.reshape((x*x*11, 4)).T

		# cv2.imshow('image',chelsea)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

	
	return np.concatenate((priors[8], priors[6], priors[4], priors[3], priors[2]), axis=1)
	


# chelsea = cv2.resize(data.chelsea(), (299, 299))
# priors = getPriors()

# i = 0
# while i < 1419:
# 	if i < 1390:
# 		i +=1
# 		continue
# 	r1, r2, c1, c2 = map(int, priors[:,i]*299.0)
# 	cv2.rectangle(chelsea, (r1, c1), (r2, c2), random_color())
# 	i += 1

# cv2.imshow('image',chelsea)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

