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
import argparse
import math

DATASET = "MNISTdata_1.hdf5"
HIDDEN_DIM = 130

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
def softmax(Z):
	return np.exp(Z) / np.sum(np.exp(Z))


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
	
	def set_data(self, X, Y):
		self.X = X
		self.Y = Y
	def set_test(self, X, Y):
		self.XT = X
		self.YT = Y


	def get_lr(self, e):
		if e < 5:
			return 0.01
		if e < 8:
			return 0.001
		return 0.0001

	"""
		Init the weights randomly (normal dist) (bias init as 0)
	"""
	def init_params(self):
		self.W  = np.random.randn(self.d_hidden, self.d) * np.sqrt(2/self.d)
		self.b1 = np.zeros((self.d_hidden, 1))
		self.b2 = np.zeros((self.k, 1))
		self.H  = np.random.randn(self.d_hidden, 1) * np.sqrt(self.d_hidden)
		self.C  = np.random.randn(self.k, self.d_hidden) * np.sqrt(2 / self.d_hidden)


	"""
		Train the model, using sgd
	"""
	def train(self, epochs=1, method="sgd"):
		N = self.X.shape[0]

		for e in range(epochs):
			lr = self.get_lr(e)
			idx = np.random.permutation(self.X.shape[0])
			for n, i in enumerate(idx):
			#for n in range(N):
				#i = np.random.randint(0,N)
				if n % 1000 == 0:
					print(f"\rProgress: [{'='*(n//1000 + 1)}{' '*(N//1000 - (n//1000+1))}]", end="")
				x = self.X[i].reshape(-1,1)
				y = self.Y[i]
				out = self.forward(x)
				dPdW, dPdB1, dPdB2, dPdC, Sigma = self.backpropagate(x, y, out)
				self.update_params(dPdW, dPdB1, dPdB2, dPdC, Sigma, lr)
			acc = self.test(self.X, self.Y)
			acc2 = self.test(self.XT, self.YT)
			print(f"\nEpoch {e+1}/{epochs} done!, Training Accuracy: {acc}, Test= {acc2} - {lr}")

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
		dPdU = out
		dPdU[y]  -= 1
		dPdB2 = dPdU
		dPdC  = np.dot(dPdU, self.H.T)
		Sigma = np.dot(self.C.T, dPdU)
		dPdB1 = Sigma * back_sigmoid(self.Z)
		dPdW  = np.dot(dPdB1, x.T) 
		return dPdW, dPdB1, dPdB2, dPdC, Sigma


	"""
		Update params by taking a step in the opposite direction of the gradients.
	"""
	def update_params(self, dPdW, dPdB1, dPdB2, dPdC, Sigma, lr):
		self.W  -= lr * dPdW
		self.b1 -= lr * dPdB1
		self.b2 -= lr * dPdB2
		self.C  -= lr * dPdC
		self.H  -= lr * Sigma

	def save(self):
		np.save("model", np.array((self.W, self.b1, self.b2, self.C)))

	def load(self, model_name):
		m = np.load(model_name, allow_pickle=True)
		self.W  = m[0]
		self.b1 = m[1]
		self.b2 = m[2]
		self.C  = m[3]

	def test(self, X, Y):
		out = self.forward(X.T)
		pred = np.argmax(out, axis=0)
		return np.mean(np.int32(pred.reshape(-1, 1) == Y))


def main():
	p = argparse.ArgumentParser(description="Neural Network")
	p.add_argument("mode", type=str, default="train", choices=("train", "load"), help="Run NN training or load exisiting model")
	p.add_argument("--hidden", default=100, help="Number of hidden layers", dest="hidden")
	p.add_argument("--epochs", default=10, help="Number of epochs", dest="epochs")
	p.add_argument("--data", default="MNISTdata_1.hdf5", help="Dataset destination", dest="data")
	p.add_argument("--model", default="model.npy", help=".npy model destination", dest="model")
	args = p.parse_args()

	xtrain, ytrain, xtest, ytest = read_data(args.data)
	K = len(set(ytrain.flatten()))
	d = xtrain.shape[1]

	nn = NeuralNet(hiddenDim=args.hidden, outputDim=K, inputDim=d)
	nn.set_data(xtrain, ytrain)
	nn.set_test(xtest, ytest)

	if args.mode == "train":
		nn.train(epochs=args.epochs)
		nn.save()
	elif args.mode == "load":
		nn.load(p.model)
	
	acc = nn.test(xtest, ytest)

	print(f"Accuracy: {acc}")
	

if __name__ == "__main__":
	main()