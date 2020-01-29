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

DATASET = "MNISTdata_1.hdf5"

def relu(Z):
	return Z * (Z>0)

def back_relu(Z):
	return Z>0
	
def sigmoid(Z):
	return np.exp(Z) / (np.exp(Z) + 1)

def back_sigmoid(Z):
	s = sigmoid(Z)
	return s * (1-s)


def softmax(Z,k=None):
	if k == None:
		return np.exp(Z) / np.sum(np.exp(Z))
	return np.exp(Z[k]) / np.sum(np.exp(Z))


def read_data(filename):
	with h5py.File(filename, "r") as data_file:
		return data_file["x_train"][:], data_file["y_train"][:], data_file["x_test"][:], data_file["y_test"][:]

def visualize(sample, pred):
	dim = np.sqrt(sample.shape[0])
	sq = sample.reshape(dim, dim)
	plt.imshow(sq)
	print(pred)
	plt.show()


def getLR(iter):
	return 0.1


class NeuralNet:
	def __init__(self, hiddenDim=0, outputDim=0, inputDim=0):
		self.MAX_ITERS = 15000
		self.d_hidden = hiddenDim
		self.k = outputDim
		self.d = inputDim
		self.init_params()
	
	def set_data(self, data):
		self.D = data

	def init_params(self):
		self.W = np.random.randn(self.d_hidden, self.d)
		self.b1 = np.random.randn(self.d_hidden, 1)
		self.b2 = np.random.randn(self.k, 1)
		self.H = np.random.randn(self.d_hidden, 1)
		self.C = np.random.randn(self.k, self.d_hidden)


	def train(self):
		for n in range(self.MAX_ITERS):
			i = random.randint(0, self.D.shape[0]-1)
			sample = self.D[i]
			x = sample[:-1].reshape(-1, 1)
			y = int(sample[-1])
			out = self.forward(x)
			self.backpropagate(x, y, out)


	def forward(self, x):
		self.Z = np.dot(self.W, x) + self.b1
		self.H = sigmoid(self.Z)
		self.U = np.dot(self.C, self.H) + self.b2
		return softmax(self.U)

	def backpropagate(self, x, y, out):
		e = np.zeros(out.shape)
		e[y] = 1
		dPdU = -(e - out)
		dPdB2 = dPdU
		dPdC = np.dot(dPdU, self.H.T)
		Sigma = np.dot(self.C.T, dPdU)
		dPdB1 = np.multiply(Sigma, back_sigmoid(self.Z))
		dPdW = np.dot(np.multiply(Sigma, back_relu(self.Z)), x.T) 
		self.update_params(dPdW, dPdB1, dPdB2, dPdC, Sigma)

	def update_params(self, dPdW, dPdB1, dPdB2, dPdC, Sigma):
		# print(dPdW.shape, self.W.shape)
		# print(dPdB1.shape, self.b1.shape)
		# print(dPdB2.shape, self.b2.shape)
		# print(dPdC.shape, self.C.shape)
		# print(Sigma.shape, self.H.shape)


		self.W -= 0.01 * dPdW
		self.b1 -= 0.01 * dPdB1
		self.b2 -= 0.01 * dPdB2
		self.C -= 0.01 * dPdC
		self.H -= 0.01 * Sigma

	def loss_func(self):
		pass


def main():
	xtrain, ytrain, xtest, ytest = read_data(DATASET)
	print(xtrain.shape)
	print(ytrain.shape)
	data = np.concatenate((xtrain, ytrain), axis=1)
	nn = NeuralNet(hiddenDim=80, outputDim=10, inputDim=xtrain.shape[1])
	nn.set_data(data)
	nn.train()
	match = 0
	for n, x in enumerate(xtest):
		out = nn.forward(x.reshape(-1, 1))
		if np.argmax(out) == ytest[n]:
			match += 1
	print(match/xtest.shape[0])

if __name__ == "__main__":
	main()