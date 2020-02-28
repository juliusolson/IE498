import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


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

EPOCHS = 10
LR = 0.001
DATASET = ""


f = h5py.File(DATASET)
x_train = f["X_train"][:]
y_train = f["Y_train"][:]
x_test = f["X_test"][:]
y_test = f["Y_test"][:]

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# Convolutional layers
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1, padding=2)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)

		self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
		self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)


		# Fully connected
		self.fc1 = nn.Linear(in_features=4096, out_features=500)
		self.fc2 = nn.Linear(in_features=500, out_features=500) # Set out = Num classes

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
		self.dropout4 = nn.Dropout(0.1)

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
		out = out.view(-1, x.size(0)) # Flatten
		out = F.relu(self.fc1(out))
		# Layer 10
		out = F.relu(self.fc2(out))
		return out



model = Net()
print(model)
	


for e in range(EPOCHS):
	model.train()

	"""
		Training
	"""


	model.eval()

	"""
		EVAL
	"""

