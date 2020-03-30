"""
	Julius Olson
	IE498 Spring 2020

	Homework 4
	-- CIFAR100 Distributed Sync SGD --

	Parts of the code was adopted from the provided examples
"""

import torch
import torch.utils.data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI
from resnet import ResNet as ResNet
from resnet import BasicBlock as BasicBlock
import numpy as np

# Code for iniitialization pytorch distributed 
print("Dist training")
cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor


# Your code start here 

"""
 FOR server 0 - K-1 (Parallel) 
	FOR epoch 0 - N-1 
		FOR EACH minibatch 
			* Calc gradient each server
			* Allreduce to sync gradient of all servers
			* Update weight of model
		END
	END
END
"""

def evaluate(loader):
	batch_acc = []
	for x, y in loader:
		x, y = Variable(x).cuda(), Variable(y).cuda()
		out = model(x)
		prediction = out.max(1)[1]
		batch_acc.append(float(prediction.eq(y).sum()) / float(BATCH_SIZE))
	acc = sum(batch_acc)/len(batch_acc)
	return acc

NUM_BLOCKS = [2,4,4,2]
CLASSES = 100
NUM_EPOCHS = 30
BATCH_SIZE = 128
DATAROOT = "/u/eot/juliusolson/scratch/data"
#DATAROOT = "./data"
LR = 5e-4

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(root=DATAROOT, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


model = ResNet(BasicBlock, NUM_BLOCKS, CLASSES, 1024)

for param in model.parameters():
	tensor0 = param.data
	dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
	param.data = tensor0/np.sqrt(np.float(num_nodes))

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
	model.train()
	for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
		X_train_batch,Y_train_batch = Variable(X_train_batch).cuda() ,Variable(Y_train_batch).cuda()
		optimizer.zero_grad()
		outputs = model(X_train_batch)
		loss = criterion(outputs, Y_train_batch)
		loss.backward()

		for param in model.parameters():
			tensor0 = param.grad.data.cpu()
			dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
			tensor0 /= float(num_nodes)
			param.grad.data = tensor0.cuda()

		optimizer.step()

		#print(f"\r Epoch {epoch+1} of {NUM_EPOCHS}. Batch {batch_idx} of {len(trainset) // BATCH_SIZE}", end="")
	#print(f"Average Loss {epoch_loss/len(trainloader)}")
	train_acc = evaluate(trainloader)
	test_acc = evaluate(testloader)

	print("Train accuracy", train_acc)
	print("Test accuracy", test_acc)
