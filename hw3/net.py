import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py
import numpy as np


EPOCHS = 15
LR = 0.0005
#DATASET = "/projects/eot/bbby/CIFAR10.hdf5"
DATASET = "CIFAR10.hdf5"
BATCH_SIZE = 128
MONTE_CARLO = False

#device = "cuda" if torch.cuda.is_available else "cpu"

class CIFAR(torch.utils.data.Dataset):
	def __init__(self, filename, training=True, transforms=None):
		f = h5py.File(filename, "r")
		
		if training:
			self.x = np.float32(f["X_train"][:])
			self.y = np.int32(f["Y_train"][:])
		else:
			self.x = np.float32(f["X_test"][:])
			self.y = np.int32(f["Y_test"][:])

	def __getitem__(self, index):
		img = self.x[index]
		label = self.y[index]
		# if transforms:
		return img, label

	def __len__(self):
		return len(self.y)
	

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

def evaluate(loader, monte_carlo=False):
	batch_acc = []
	for x, y in loader:
		x = Variable(x).cuda()
		y = Variable(y.long()).cuda()
		
		out = 0.0
		if monte_carlo:
			for i in range(10):
				out += model(x)/10
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

train_data = CIFAR(DATASET, training=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = CIFAR(DATASET, training=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


"""
	Training
"""
print("Training")


for e in range(EPOCHS):
	model.train()
	for i, (x, y) in enumerate(train_loader):
		x = Variable(x).cuda()
		y = Variable(y.long()).cuda()

		outputs = model(x)
		loss = criterion(outputs, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print(f"\rEpoch {e+1}. Batch {i} of {len(train_data)//BATCH_SIZE}, Loss: {loss}", end="")

	if e == EPOCHS-1:
		torch.save({
			"epoch": e,
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"loss": loss,
		}, "checkpoint.pt")

	print("")
	if not MONTE_CARLO:
		model.eval() # model.eval defaults to using heuristic

	train_acc = evaluate(train_loader, monte_carlo=MONTE_CARLO)
	test_acc = evaluate(test_loader, monte_carlo=MONTE_CARLO)
	
	
	
	print("Train Accuarcy: ", train_acc, "Test Accuracy: ", test_acc)

