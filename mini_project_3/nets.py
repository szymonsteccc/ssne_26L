import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from tqdm import tqdm



class Net(nn.Module):
    def __init__(self, kernel_sizes=[5, 2, 2], n_channels=6, num_classes=50, hidden_size=25):
        super().__init__()
        ## Warstwa konwolucyjna
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=kernel_sizes[0], stride=1, padding=0)
        ## Warstwa max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=kernel_sizes[1], stride=2)
        self.conv2 = nn.Conv2d(n_channels, 16, kernel_size=kernel_sizes[2])
        self.pool2 = nn.MaxPool2d(kernel_size=kernel_sizes[2], stride=2)
        # !!!
        self.fc1 = nn.Linear(3136, hidden_size)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # !!!
        x = torch.flatten(x, start_dim=1) # flatten all dimensions except batch
        # print(x.size())
        # !!!
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# skopiowane z lab6:
# trzeba zamineć kernel_size na kernel_sizes i dodać hidden_size do argumentów konstruktora, a potem przekazać je do warstwy fc1, która musi być LazyLinear, bo nie znamy rozmiaru wejścia (zależy on od rozmiaru obrazu i kernel_sizes)
class LongNet(nn.Module):
	def __init__(self, num_classes=50, hidden_size=[5, 10]):
		super().__init__()
		self.n_convs = 10
		n_channels = 10

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=5, stride=1, padding=0)
		self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, stride=1, padding=0)
		self.convs = nn.ModuleList(
			[nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels, kernel_size=5, stride=1, padding=2) for _ in range(self.n_convs)]
		)
		self.bns = nn.ModuleList([nn.BatchNorm2d(n_channels) for _ in range(self.n_convs)])

		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fc1 = nn.LazyLinear(hidden_size[0])
		self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
		self.fc3 = nn.Linear(hidden_size[1], num_classes)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		res = F.relu(self.conv2(x))
		x = res
		for i in range(self.n_convs):
			x = torch.cat([x, res], dim=1)
			x = F.relu(self.convs[i](x))
			x = self.bns[i](x)
		x = self.pool1(x)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
