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
    def __init__(self, num_classes=50):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
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
