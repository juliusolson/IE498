"""
	Julius Olson
	IE498 Spring 2020

	Homework 1
	-- Neural Net from scratch in Numpy --
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt
import random
from sys import argv
import sys

DATASET = "MNISTdata_1.hdf5"

""" 
	Activation functions
	Relu and sigmoid (forward and backward)
"""

def relu(Z):
	return Z * (Z>0) # max (Z, 0)

def back_relu(Z):
	return Z>0 # 1 if Z> 0 , else 0
	
def sigmoid(Z):
	return np.exp(Z) / (np.exp(Z) + 1)

def back_sigmoid(Z):
	s = sigmoid(Z)
	return s * (1-s)


"""
	Softmax returns distribution
"""
def softmax(Z,k=None):
	if k == None:
		return np.exp(Z) / np.sum(np.exp(Z))
	return np.exp(Z[k]) / np.sum(np.exp(Z))


def read_data(filename):
	with h5py.File(filename, "r") as data_file:
		return data_file["x_train"][:], data_file["y_train"][:], data_file["x_test"][:], data_file["y_test"][:]


"""
 Visualizes a samples as an image
"""
def visualize(sample, pred):
	dim = int(np.sqrt(sample.shape[0]))
	sq = sample.reshape(dim, dim)
	plt.imshow(sq)
	print(pred)
	plt.show()

class NeuralNet:
	def __init__(self, hiddenDim=0, outputDim=0, inputDim=0):
		self.d_hidden = hiddenDim
		self.k = outputDim
		self.d = inputDim
		self.init_params()
	
	def set_data(self, data):
		self.D = data


	def get_lr(self, i):
		return 0.01
		# if i < 10e3:
		# 	return 0.1
		# elif i < 10e4:
		# 	return 0.01
		# elif i < 10e5:
		# 	return 0.001
		# return 0.0001

	"""
		Init the params randomly (normal dist)
	"""
	def init_params(self):
		self.W = np.random.randn(self.d_hidden, self.d)
		self.b1 = np.random.randn(self.d_hidden, 1)
		self.b2 = np.random.randn(self.k, 1)
		self.H = np.random.randn(self.d_hidden, 1)
		self.C = np.random.randn(self.k, self.d_hidden)


	"""
		Train the model, by sgd
	"""
	def train(self, epochs=1, method="sgd"):
		for e in range(epochs):
			for n in range(self.D.shape[0]):
				if n % 1000 == 0:
					print(f"\r Progress: {n} / {self.D.shape[0]}", end="")
				i = random.randint(0, self.D.shape[0]-1)
				sample = self.D[i]
				x = sample[:-1].reshape(-1, 1)
				y = int(sample[-1])
				out = self.forward(x)
				dPdW, dPdB1, dPdB2, dPdC, Sigma = self.backpropagate(x, y, out)
				lr = self.get_lr((e+1)*n)
				self.update_params(dPdW, dPdB1, dPdB2, dPdC, Sigma, lr)
			print("Epoch Done")


	"""
		Forward propagate
	"""
	def forward(self, x):
		self.Z = np.dot(self.W, x) + self.b1
		self.H = sigmoid(self.Z)
		self.U = np.dot(self.C, self.H) + self.b2
		return softmax(self.U)


	"""
		Calculate the gradients
	"""
	def backpropagate(self, x, y, out):
		e = np.zeros(out.shape)
		e[y] = 1
		dPdU = -(e - out)
		dPdB2 = dPdU
		dPdC = np.dot(dPdU, self.H.T)
		Sigma = np.dot(self.C.T, dPdU)
		dPdB1 = np.multiply(Sigma, back_sigmoid(self.Z))
		dPdW = np.dot(dPdB1, x.T) 
		return dPdW, dPdB1, dPdB2, dPdC, Sigma


	"""
		Update params by taking a step in the opposite direction of the gradients.
	"""
	def update_params(self, dPdW, dPdB1, dPdB2, dPdC, Sigma, lr):
		self.W -= lr * dPdW
		self.b1 -= lr * dPdB1
		self.b2 -= lr * dPdB2
		self.C -= lr * dPdC
		self.H -= lr * Sigma

	def save(self):
		np.save("model", np.array((self.W, self.b1, self.b2, self.C)))

	def load(self):
		m = np.load("model.npy", allow_pickle=True)
		self.W = m[0]
		self.b1 = m[1]
		self.b2 = m[2]
		self.C = m[3]

	def test(self, X, Y):
		out = self.forward(X.T)
		pred = np.argmax(out, axis=0)
		return np.mean(np.int32(pred.reshape(-1, 1) == Y))


def main():
	if len(argv) < 2:
		print("Need args...")
		sys.exit(0)

	xtrain, ytrain, xtest, ytest = read_data(DATASET)
	print(xtrain.shape)
	print(ytrain.shape)
	data = np.concatenate((xtrain, ytrain), axis=1)
	nn = NeuralNet(hiddenDim=100, outputDim=10, inputDim=xtrain.shape[1])
	nn.set_data(data)
	
	
	if argv[1] == "train":
		nn.train(epochs=10)
		nn.save()
	elif argv[1] == "load":
		nn.load()
	
	acc = nn.test(xtest, ytest)

	print(f"Accuracy: {acc}")
	

if __name__ == "__main__":
	main()