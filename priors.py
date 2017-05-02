import numpy as np 

def getPriors():
	IMAGE_WIDTH = 299
	IMAGE_HEIGHT = 299
	gridDim = np.array([8, 6, 4, 3, 2])
	numPriors = np.square(gridDim) * 11

	aspectRatioW = np.linspace(start=2, stop=3, num=11)
	aspectRatioH = np.linspace(start=2, stop=1, num=11)

	priors = {}

	for x in gridDim:
		block = np.zeros((x, x, 44))
		
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
					d += 1
		priors[x] = block.reshape((4, x*x*11))

	
	return np.concatenate((priors[8], priors[6], priors[4], priors[3], priors[2]), axis=1)
	
# import cv2
# from skimage import data

# chelsea = cv2.resize(data.chelsea(), (299, 299))

# def random_color():
#     return np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)

# for key in priors:
# 	for row in priors[2]:
# 		d = 0
# 		while d < 11:
# 			r1, c1, r2, c2 = map(int, row[4*d:4*d+4]*299.0)
# 			cv2.rectangle(chelsea, (r1, c1), (r2, c2), random_color())
# 			d += 1
# cv2.imshow('image',chelsea)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

