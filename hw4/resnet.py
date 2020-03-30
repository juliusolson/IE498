"""
	Julius Olson
	IE498 Spring 2020

	Homework 4
	-- RenNet and BasicBlock Classes --

	Parts of the code was adopted from the provided examples
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=False):
		super(BasicBlock, self).__init__()

		self.relu = nn.ReLU(inplace=True)		
		self.downsample = downsample
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
			nn.BatchNorm2d(out_channels),
			self.relu,
		)		
		self.conv2 = nn.Sequential(
			nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(out_channels),
		)
		if self.downsample:
			self.proj = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.conv2(out)

		if self.downsample:
			residual = self.proj(x)
		out += residual
		out = self.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, basic_block, num_blocks, num_classes, fc_in):
		super(ResNet, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.15)
		)
		
		self.conv2 = self.make_layer(basic_block, 32, 32, num_blocks[0])
		self.conv3 = self.make_layer(basic_block, 32, 64, num_blocks[1], stride=2)
		self.conv4 = self.make_layer(basic_block, 64, 128, num_blocks[2], stride=2)
		self.conv5 = self.make_layer(basic_block, 128, 256, num_blocks[3], stride=2)
		
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(fc_in, num_classes)

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.pool(out)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)
		return out

	def make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
		downsample = True if stride > 1 else False
		layers = [ block(in_channels, out_channels, stride=stride, downsample=downsample) ]
		for i in range(1, num_blocks):
			layers.append(block(out_channels, out_channels))
		return nn.Sequential(*layers)