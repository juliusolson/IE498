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
import time

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


# def conv(x, k):
# 	k_x, k_y = k.shape
# 	d = x.shape[0]
# 	out = np.zeros((d-k_y+1, d-k_x+1))
# 	for i in range(out.shape[0]):
# 		for j in range(out.shape[1]):
# 			out[i,j] = np.sum(np.multiply(k, x[i:i+k_y, j:j+k_x]))
# 	return out

def vec_conv(x, k):
	k_y, k_x, k_ch = k.shape
	d_out = x.shape[0] - k_y + 1

	k_matrix = np.zeros((k_ch, k_y*k_x))
	for p in range(k_ch):
		k_matrix[p, :] = k[:,:,p].flatten()
	
	x_matrix = np.zeros((k_y*k_x, d_out*d_out))
	col = 0
	for i in range(d_out):
		for j in range(d_out):
			x_matrix[:, col] = x[i:i+k_y, j:j+k_x].flatten()
			col += 1
	
	return np.dot(k_matrix, x_matrix).T.reshape(d_out, d_out, k_ch)


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
		self.classes = outputDim
		self.k_dim = filter_size
		self.d = inputDim
		self.Ch = number_channels
		self.lr = 0.001
		self.init_params()
	
	"""
		Init the weights and bias
	"""
	def init_params(self):
		self.W = np.random.randn(self.classes, self.d-self.k_dim+1, self.d-self.k_dim+1, self.Ch) * np.sqrt(2 / (self.d * self.d))
		self.K = np.random.randn(self.k_dim, self.k_dim, self.Ch) * np.sqrt(2 / (self.k_dim * self.k_dim))
		self.b = np.zeros((self.classes, 1))
	
	def set_data(self, X, Y):
		self.X = X
		self.Y = Y

	"""
		Train the model, using sgd
	"""
	def train(self, epochs=1):
		N = self.X.shape[0]

		for e in range(epochs):
			start = time.time()
			lr = 0.01
			idx = np.random.permutation(self.X.shape[0])
			for n, i in enumerate(idx):
				if n % 1000 == 0:
				   	print(f"\rProgress: [{'='*(n//1000 + 1)}{' '*(N//1000 - (n//1000+1))}]", end="")
				x = self.X[i].reshape(self.d,self.d)
				y = self.Y[i]
				out = self.forward(x)
				dPdB, dPdK, dPdW = self.backpropagate(x, y, out)
				self.update_params(dPdB, dPdK, dPdW, lr)
			end = time.time()
			acc = self.test(self.X, self.Y)
			print(f"\nEpoch {e+1}/{epochs} done!, Training Accuracy: {acc}, Time: {end-start}")

	"""
		Forward propagate
	"""
	def forward(self, x):
		self.Z = vec_conv(x, self.K)
		self.H = relu(self.Z)

		self.U = np.zeros((self.classes, 1))
		for k in range(self.classes):
			self.U[k] = np.sum(np.multiply(self.W[k,:,:,:], self.H))
		self.U += self.b
		return softmax(self.U)

	"""
		Calculate the gradients
	"""
	def backpropagate(self, x, y, out):
		dPdU = np.zeros(out.shape)
		dPdU[y] = -1
		dPdU += out
		Delta = np.tensordot(dPdU.flatten(), self.W, axes=((0), (0)))
		dPdW = np.zeros(self.W.shape)
		for k in range(self.classes):
			dPdW[k] = dPdU[k] * self.H
		dPdK = vec_conv(x, np.multiply(back_relu(self.Z), Delta ))
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
	def save(self, epochs):
		np.save("model", np.array((self.W, self.K, self.b, epochs)))
	"""
		Load nn from .npy file
	"""
	def load(self, model_name):
		m = np.load(model_name, allow_pickle=True)
		self.W = m[0]
		self.K = m[1]
		self.b = m[2]
		try:
			epochs = m[3]
		except: 
			print("No epoch info")
			return -1
		return epochs
		

	"""
		Evaulate on test set
	"""
	def test(self, X, Y):
		# out = np.zeros((X.shape[0], self.classes))
		# for i in range(X.shape[0]):
		# 	out[i] = self.forward(X[i].reshape(28,28)).flatten()
		# pred = np.argmax(out, axis=1)
		# return np.mean(np.int32(pred.reshape(-1, 1) == Y))

		print(X.shape[0])
		tot = X.shape[0]
		corr = 0
		for i in range(X.shape[0]):
			out = self.forward(X[i].reshape(28,28)).flatten()
			pred = np.argmax(out)
			corr += (int(pred) == Y[i])
		return corr / tot

def main():
	# Tweak settings from command line
	p = argparse.ArgumentParser(description="Neural Network")
	p.add_argument("mode", type=str, default="train", choices=("train", "load"), help="Run NN training or load exisiting model")
	p.add_argument("--channels", default=3, help="Number of channels", dest="channels")
	p.add_argument("--filter", default=7, help="Filter dimensions", dest="filter_dim")
	p.add_argument("--epochs", default=10, help="Number of epochs", dest="epochs")
	p.add_argument("--data", default="MNISTdata_1.hdf5", help="Dataset destination", dest="data")
	p.add_argument("--model", default="model.npy", help=".npy model destination", dest="model")
	args = p.parse_args()

	# Read data and set output (K) and input (d) dimensions
	xtrain, ytrain, xtest, ytest = read_data(args.data)
	K = len(set(ytrain.flatten()))
	d = int(np.sqrt(xtrain.shape[1]))

	# Init model
	nn = NeuralNet(
		outputDim=K, 
		inputDim=d,
		filter_size=int(args.filter_dim), 
		number_channels=int(args.channels)
	)

	nn.set_data(xtrain, ytrain)


	# Either train or load prev. model
	if args.mode == "train":
		nn.train(epochs=int(args.epochs))
		nn.save(int(args.epochs))
	elif args.mode == "load":
		epochs = nn.load(args.model)
		print(f"Loaded model: W dim: {nn.W.shape}, K dim: {nn.K.shape}, Epochs: {epochs}")
	
	# Calculate test accuracy
	acc = nn.test(xtest, ytest)

	print(f"Test accuracy: {acc}")
	

if __name__ == "__main__":
	main()
