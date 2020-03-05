import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import h5py
import numpy as np


"""

init Architecture

1. Conv (channels=64, k-dim=4, stride=1, Pad=2)
	-- Batch normalization 1
2. Conv (c=64, k=4, s=1, P=2)
	-- Max Pooling: s=2, k=2
	-- Dropout 1
3. Conv (ch=64, k=4, s=1, P=2)
	-- Batch norm 2
4. Conv (ch=64, k=4, s=1, P=2)
	-- Maxpooling 
	-- Dropout 2
5. Conv (ch=64, k=4, s=1, P=2)
	-- Batch norm. 3
6. Conv (ch=64, k=3, s=1, P=0)
	-- Dropout 3
7. Conv (ch=64, k=3, s=1, P=0)
	-- Batch norm, 4
8. Conv (ch=64, k=3, s=1, P=0)
	-- Batch norm. 5
	-- Dropout 4
9. FC (500 units)
10. FC (500 units)
11. Linear => Softmax 

"""

EPOCHS = 20
LR = 0.001
DATASET = "/projects/eot/bbby/CIFAR10.hdf5"
#DATASET = "CIFAR10.hdf5"
BATCH_SIZE = 64


#device = "cuda" if torch.cuda.is_available else "cpu"


print("Loading Data...")
f = h5py.File(DATASET, "r")
x_train = np.float32(f["X_train"][:])
y_train = np.int32(f["Y_train"][:])
x_test = np.float32(f["X_test"][:])
y_test = np.int32(f["Y_test"][:])


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# Convolutional layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=(2, 2))
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=(2, 2))
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=(2, 2))
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=(2, 2))
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=(2, 2))

		self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)


		# Fully connected
		self.fc1 = nn.Linear(in_features=1024, out_features=512)
		self.fc2 = nn.Linear(in_features=512, out_features=256) # Set out = Num classes
		self.fc3 = nn.Linear(in_features=256, out_features=10)

		# Batch Normalization
		self.Bnorm1 = nn.BatchNorm2d(64)
		self.Bnorm2 = nn.BatchNorm2d(64)
		self.Bnorm3 = nn.BatchNorm2d(64)
		self.Bnorm4 = nn.BatchNorm2d(64)
		self.Bnorm5 = nn.BatchNorm2d(64)

		# Dropouts
		self.dropout1 = nn.Dropout(0.1)
		self.dropout2 = nn.Dropout(0.1)
		self.dropout3 = nn.Dropout(0.1)
		self.dropout4 = nn.Dropout(0.2)

		# Maxpool
		self.pool = nn.MaxPool2d(2,2)



	def forward(self, x):
		# Layer1
		out = self.Bnorm1(F.relu(self.conv1(x)))
		# Layer2
		out = F.relu(self.conv2(out))
		p = self.dropout1(self.pool(out))
		# Layer 3
		out = self.Bnorm2(F.relu(self.conv3(p)))
		# Layer 4
		out = F.relu(self.conv4(out))
		p = self.dropout2(self.pool(out))
		# Layer 5
		out = self.Bnorm3(F.relu(self.conv5(p)))
		# Layer 6
		out = self.dropout3(F.relu(self.conv6(out)))
		# Layer 7
		out = self.Bnorm4(F.relu(self.conv7(out)))
		# Layer 8
		out = self.Bnorm5(F.relu(self.conv8(out)))
		out = self.dropout4(out)
		# Layer 9
		out = out.view(x.size(0), -1) # Flatten
		out = F.relu(self.fc1(out))
		# Layer 10
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out


def evaluate(data, targets):
	acc = 0.0
	batch_acc = []
	#with torch.no_grad():
	for i in range(0, len(targets), BATCH_SIZE):
		data_batch = torch.FloatTensor(data[i:i+BATCH_SIZE, :])
		target_batch = torch.LongTensor(targets[i:i+BATCH_SIZE])
		d, t = Variable(data_batch).cuda(), Variable(target_batch).cuda()
		out  = model(d)			
		prediction = out.max(1)[1]
		batch_acc.append(float(prediction.eq(t).sum()) / float(BATCH_SIZE))
	acc = sum(batch_acc) / len(batch_acc)
	return acc


model = Net()
model.cuda()

print("Model setup...")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

"""
	Training
"""
print("Training")
for e in range(EPOCHS):
	model.train()
	indices = np.random.permutation(len(y_train))

	x_train = x_train[indices, :]
	y_train = y_train[indices]

	for i in range(0, len(y_train), BATCH_SIZE):
		#x_batch = torch.FloatTensor(x_train[i:i+BATCH_SIZE, :])
		x_batch = torch.FloatTensor(x_train[i:i+BATCH_SIZE, :])
		y_batch = torch.LongTensor(y_train[i:i+BATCH_SIZE])

		x, y = Variable(x_batch).cuda(), Variable(y_batch).cuda()

		optimizer.zero_grad()
		out = model(x)
		loss = criterion(out, y)
		loss.backward()

		optimizer.step()
		#print(f"\rEpoch {e+1}. Batch {i // BATCH_SIZE} of {len(y_train)//BATCH_SIZE}, Loss: {loss}", end="")



	"""
		EVAL
	"""

	print("\nEval...")
	model.eval()
		
	train_acc = evaluate(x_train, y_train)
	test_acc = evaluate(x_test, y_test)

	#print(f"Training Accuracy: {train_acc}")
	#print(f"Test Accuracy: {test_acc}")
	print(train_acc)
	print(test_acc)

