"""
	Julius Olson
	IE498 Spring 2020

	Homework 2
	-- Convolutional Neural Net from scratch in Numpy --
"""

import numpy as np
import h5py
import argparse
from scipy import signal

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


def conv(x, k):
	k_x, k_y = k.shape
	d = x.shape[0]
	out = np.zeros((d-k_y+1, d-k_x+1))
	for i in range(out.shape[0]):
		for j in range(out.shape[1]):
			out[i,j] = np.sum(np.multiply(k, x[i:i+k_y, j:j+k_x]))
	return out


"""
	Softmax returns distribution
"""
def softmax(Z):
	return np.exp(Z) / np.sum(np.exp(Z))


def read_data(filename):
	with h5py.File(filename, "r") as data_file:
		return data_file["x_train"][:], data_file["y_train"][:], data_file["x_test"][:], data_file["y_test"][:]
		

class NeuralNet:
	def __init__(self, outputDim=0, inputDim=0, filter_size=0, number_channels=0):
		self.k = outputDim
		self.k_dim = filter_size
		self.d = inputDim
		self.Ch = number_channels
		self.init_params()
	
	"""
		Init the weights and bias
	"""
	def init_params(self):
		self.W = np.random.randn(self.k, self.Ch, self.d-self.k_dim+1, self.d-self.k_dim+1)
		self.K = np.random.randn(self.Ch, self.k_dim, self.k_dim)
		self.b = np.random.randn(self.k, 1)
	
	def set_data(self, X, Y):
		self.X = X
		self.Y = Y

	def get_lr(self, e):
		if e < 1:
			return 0.1
		if e < 5:
			return 0.01
		if e < 8:
			return 0.001
		return 0.0001

	"""
		Train the model, using sgd
	"""
	def train(self, epochs=1):
		N = self.X.shape[0]

		for e in range(epochs):
			lr = 0.01
			idx = np.random.permutation(self.X.shape[0])
			for n, i in enumerate(idx):
				#print(f"\r{n}", end="")
				if n % 1000 == 0:
				   	print(f"\rProgress: [{'='*(n//1000 + 1)}{' '*(N//1000 - (n//1000+1))}]", end="")
				x = self.X[i].reshape(self.d,self.d)
				y = self.Y[i]
				out = self.forward(x)
				dPdB, dPdK, dPdW = self.backpropagate(x, y, out)
				self.update_params(dPdB, dPdK, dPdW, lr)
			acc = self.test(self.X, self.Y)
			print(f"\nEpoch {e+1}/{epochs} done!, Training Accuracy: {acc}")

	"""
		Forward propagate
	"""
	def forward(self, x):
		self.Z = np.zeros((self.Ch, self.d-self.k_dim+1, self.d-self.k_dim+1))
		for c in range(self.Ch):
			self.Z[c] = signal.convolve(x, self.K[c],mode="valid")
		self.H = sigmoid(self.Z)
		
		# (K C I J) X (C I J) => Reduce axis CIJ i.e. 1.2.3 and 0,1,2 respectively
		self.U = np.tensordot(self.W, self.H, axes=((1,2,3), (0,1,2)))
		self.U = self.U.reshape(-1,1) + self.b
		return softmax(self.U+self.b)


	"""
		Calculate the gradients
	"""
	def backpropagate(self, x, y, out):
		dPdU = np.zeros(out.shape)
		dPdU[y] = -1
		dPdU += out

		dPdW = np.zeros(self.W.shape)
		for k in range(self.k):
			dPdW[k] = dPdU[k]*self.H

		# Eliminate the first axis of dPdU and W via summation
		Delta = np.tensordot(dPdU.flatten(), self.W, axes=((0),(0)))
		dPdK = np.zeros((self.Ch, self.k_dim, self.k_dim))
		for c in range(self.Ch):
			dPdK[c] = signal.convolve(x, np.multiply(back_sigmoid(self.Z[c,:,:]), Delta[c,:,:]), mode="valid")
		return dPdU, dPdK, dPdW





	"""
		Update params by taking a step in the opposite direction of the gradients.
	"""
	def update_params(self, dPdB, dPdK, dPdW, lr):
		self.b -= lr * dPdB
		self.W -= lr * dPdW
		self.K -= lr * dPdK

	"""
		Save the model to .npy file
	"""
	def save(self):
		#np.save("model", np.array((self.W, self.b1, self.b2, self.C)))
		pass
	"""
		Load nn from .npy file
	"""
	def load(self, model_name):
		m = np.load(model_name, allow_pickle=True)
		

	"""
		Evaulate on test set
	"""
	def test(self, X, Y):
		out = np.zeros((X.shape[0], self.k))
		for i in range(X.shape[0]):
			out[i] = self.forward(X[i].reshape(28,28)).flatten()
		pred = np.argmax(out, axis=1)
		return np.mean(np.int32(pred.reshape(-1, 1) == Y))


def main():
	# Tweak settings from command line
	p = argparse.ArgumentParser(description="Neural Network")
	p.add_argument("mode", type=str, default="train", choices=("train", "load"), help="Run NN training or load exisiting model")
	p.add_argument("--channels", default=3, help="Number of channels", dest="channels")
	p.add_argument("--filter", default=3, help="Filter dimensions", dest="filter_dim")
	p.add_argument("--epochs", default=10, help="Number of epochs", dest="epochs")
	p.add_argument("--data", default="MNISTdata_1.hdf5", help="Dataset destination", dest="data")
	p.add_argument("--model", default="model.npy", help=".npy model destination", dest="model")
	args = p.parse_args()

	# Read data and set output (K) and input (d) dimensions
	xtrain, ytrain, xtest, ytest = read_data(args.data)
	K = len(set(ytrain.flatten()))
	d = xtrain.shape[1]

	# Init model
	nn = NeuralNet(
		outputDim=K, 
		inputDim=int(np.sqrt(d)),
		filter_size=int(args.filter_dim), 
		number_channels=int(args.channels)
	)

	nn.set_data(xtrain, ytrain)


	# Either train or load prev. model
	if args.mode == "train":
		nn.train(epochs=int(args.epochs))
		nn.save()
	elif args.mode == "load":
		nn.load(args.model)
	
	# Calculate test accuracy
	acc = nn.test(xtest, ytest)

	print(f"Test accuracy: {acc}")
	

if __name__ == "__main__":
	main()
