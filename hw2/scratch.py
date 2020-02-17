import numpy as np
import h5py
import time


def read_data(filename):
	with h5py.File(filename, "r") as data_file:
		return data_file["x_train"][:], data_file["y_train"][:], data_file["x_test"][:], data_file["y_test"][:]


filename = "MNISTdata_1.hdf5"
x_train, y_train, x_test, y_test = read_data(filename)

img = x_train[0].reshape(28, 28)
ker = np.random.rand(3, 3)

def conv(x, k):
	out = np.zeros((x.shape[0]-k.shape[0]+1, x.shape[0]-k.shape[0]+1))
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			out[i,j] = np.sum(np.multiply(k, x[i:i+k.shape[1], j:j+k.shape[0]]))
	return out


print(img.shape)

tot = 0.0
for i in range(100):
	t0 = time.time()
	out = conv(img, ker)
	t1 = time.time()
	tot+=(t1-t0)
print(tot/100)
