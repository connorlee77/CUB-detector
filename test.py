import numpy as np 

x = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

a = np.zeros((len(x), 4, 10))
for i, row in enumerate(x):
	a[i] = np.repeat(np.array([row]).T, 10, axis=1)

print(a)
