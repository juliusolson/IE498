"""
	Julius Olson
	IE498 Spring 2020

	Homework 3
	-- CIFAR10 Net using pytorch and GPU training --
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import h5py
import numpy as np


"""
	Hyperparameters
"""
EPOCHS = 15
LR = 0.0005
#DATASET = "/projects/eot/bbby/CIFAR10.hdf5"
DATASET = "CIFAR10.hdf5"
BATCH_SIZE = 128
MONTE_CARLO = False
MONTE_CARLO_ITERATIONS = 10

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv_block1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.05),
		)

		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=5, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=5, padding=2),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.05)
		)

		self.conv_block3 = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=5, padding=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=5, padding=2),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2,stride=2),
			nn.Dropout(0.1),
		)

		self.linear_block = nn.Sequential(
			nn.Linear(4096, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.ReLU(inplace=True),
			nn.Linear(256, 10),
		)

	def forward(self, x):

		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = self.conv_block3(x)

		x = x.view(x.size(0), -1)

		x = self.linear_block(x)

		return x

"""
	input:  loader: dataloader, monte_carlo: boolean
	output: the predictive accuracy for the model on the data provided by the loader
			Monte carlo is performed if defined by the parameter
"""
def evaluate(loader, monte_carlo=False):
	batch_acc = []
	for x, y in loader:
		x = Variable(x).cuda()
		y = Variable(y.long()).cuda()
		
		out = 0.0
		if monte_carlo:
			for i in range(MONTE_CARLO_ITERATIONS):
				print(f"\rMonte Carlo Sim {i}", end="")
				out += model(x)/MONTE_CARLO_ITERATIONS
			print("")
		else:
			out = model(x)

		prediction = out.max(1)[1]
		batch_acc.append(float(prediction.eq(y).sum()) / float(BATCH_SIZE))
	acc = sum(batch_acc)/len(batch_acc)
	return acc



model = Net()
model.cuda()

print("Model setup...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Loading data...")

"""
	Data Augmentation
"""
transform = transforms.Compose([
	#transforms.RandomCrop(32),
	transforms.RandomVerticalFlip(),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
])
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


"""
	Training
"""
print("Training...")

for e in range(EPOCHS):
	model.train()
	epoch_loss = 0.0
	for i, (x, y) in enumerate(train_loader):
		x = Variable(x).cuda()
		y = Variable(y.long()).cuda()

		outputs = model(x)
		loss = criterion(outputs, y)
		epoch_loss+=loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print(f"\rEpoch {e+1} of {EPOCHS}. Batch {i} of {len(train_data)//BATCH_SIZE}", end="")
	# Save model
	if e == EPOCHS-1:
		torch.save({
			"epoch": e,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"loss": loss,
		}, "checkpoint.pt")
	print("")
	
	print(f"Average Loss {epoch_loss/len(train_loader)}")
	if not MONTE_CARLO:
		model.eval() # model.eval defaults to using heuristic

	train_acc = evaluate(train_loader, monte_carlo=MONTE_CARLO)
	test_acc = evaluate(test_loader, monte_carlo=MONTE_CARLO)
	
	print("Train Accuarcy: ", train_acc, "Test Accuracy: ", test_acc)

