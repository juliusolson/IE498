"""
	Julius Olson
	IE498 Spring 2020

	Homework 4
	-- CIFAR100 Pretrained --

	Parts of the code was adopted from the provided examples
"""


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms as transforms

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Loading the data

def resnet18(pretrained=True):
	model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./model/'))
	return model

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

NUM_EPOCHS = 10
DATAROOT = "./data"
BATCH_SIZE = 128
LR = 1e-4

model = resnet18(pretrained=True)

# If you just need to fine-tune the last layer, comment out the code below.
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 100)


transform_train = transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
	transforms.Resize(224),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root=DATAROOT, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root=DATAROOT, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

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
		optimizer.step()

		print(f"\r Epoch {epoch+1} of {NUM_EPOCHS}. Batch {batch_idx} of {len(trainset) // BATCH_SIZE}", end="")
	#print(f"Average Loss {epoch_loss/len(trainloader)}")
	model.eval()
	train_acc = evaluate(trainloader)
	test_acc = evaluate(testloader)
	print("")
	print(train_acc)
	print(test_acc)
