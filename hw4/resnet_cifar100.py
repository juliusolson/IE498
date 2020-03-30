"""
	Julius Olson
	IE498 Spring 2020

	Homework 4
	-- CIFAR100 Resnet --

	Parts of the code was adopted from the provided examples
"""

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
from resnet import BasicBlock as BasicBlock
from resnet import ResNet as ResNet


CLASSES = 100
NUM_BLOCKS = [2, 4, 4, 2]
NUM_EPOCHS = 30
BATCH_SIZE = 128
#DATAROOT = "/u/eot/juliusolson/scratch/data"
DATAROOT = "./data"
LR = 0.0005

def evaluate(loader):
	batch_acc = []
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			out = model(x)
			prediction = out.max(1)[1]
			batch_acc.append(float(prediction.eq(y).sum()) / float(BATCH_SIZE))
		acc = sum(batch_acc)/len(batch_acc)
	return acc

# torch.manual_seed(0)
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root=DATAROOT, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root=DATAROOT, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet(BasicBlock, NUM_BLOCKS, CLASSES)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
	# Train the model
	model.train()
#	epoch_loss = 0.0
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
		X_train_batch,Y_train_batch = X_train_batch.to(device),Y_train_batch.to(device)
		outputs = model(X_train_batch)
		loss = criterion(outputs, Y_train_batch)
		#epoch_loss+=loss

		optimizer.zero_grad()
		loss.backward()

		if epoch > 16:
			for group in optimizer.param_groups:
				for p in group["params"]:
					state = optimizer.state[p]
					if state["step"] >= 1024:
						state["step"] = 1000
		optimizer.step()

		#print(f"\r Epoch {epoch+1} of {NUM_EPOCHS}. Batch {batch_idx} of {len(trainset) // BATCH_SIZE}", end="")
	#print(f"Average Loss {epoch_loss/len(trainloader)}")
	model.eval()
	train_acc = evaluate(trainloader)
	test_acc = evaluate(testloader)

	print(train_acc)
	print(test_acc)
