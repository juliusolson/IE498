"""
	Julius Olson
	IE498 Spring 2020

	Homework 4
	-- TinyImage ResNet --

	Parts of the code was adopted from the provided examples
"""

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from resnet import ResNet as ResNet
from resnet import BasicBlock as BasicBlock

NUM_EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4
BASE_DIR = "./data/tiny-imagenet-200"

transform_train = transforms.Compose([
	transforms.RandomCrop(64, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
])

transform_val = transforms.Compose([
	transforms.ToTensor(),
])

def create_val_folder(val_dir):
	"""
	This method is responsible for separating validation images into separate sub folders
	"""
	path = os.path.join(val_dir, 'images')  # path where validation data is present now
	filename = os.path.join(val_dir, 'val_annotations.txt')  # file where image2class mapping is present
	fp = open(filename, "r")  # open file in read mode
	data = fp.readlines()  # read line by line

	# Create a dictionary with image names as key and corresponding classes as values
	val_img_dict = {}
	for line in data:
		words = line.split("\t")
		val_img_dict[words[0]] = words[1]
	fp.close()
	# Create folder if not present, and move image into proper folder
	for img, folder in val_img_dict.items():
		newpath = (os.path.join(path, folder))
		if not os.path.exists(newpath):  # check if folder exists
			os.makedirs(newpath)
		if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
			os.rename(os.path.join(path, img), os.path.join(newpath, img))
	return

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

# Your own directory to the train folder of tiyimagenet
train_dir = BASE_DIR+"/train/"
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# To check the index for each classes
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
# Your own directory to the validation folder of tiyimagenet
val_dir = BASE_DIR+'/val/'


if 'val_' in os.listdir(val_dir+'images/')[0]:
	create_val_folder(val_dir)
	val_dir = val_dir+'images/'
else:
	val_dir = val_dir+'images/'


val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
# To check the index for each classes
# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

# -----------------------------------------
print("MODEL SETUP ")
model = ResNet(BasicBlock, [2,4,4,2], 200, 4096)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
	# Train the model
	model.train()
#	epoch_loss = 0.0
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):
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
		print(f"\r Epoch {epoch+1} of {NUM_EPOCHS}. Batch {batch_idx} of {len(train_dataset) // BATCH_SIZE}", end="")
	#print(f"Average Loss {epoch_loss/len(trainloader)}")
	model.eval()
	train_acc = evaluate(train_loader)
	val_acc = evaluate(val_loader)

	print(train_acc)
	print(val_acc)



