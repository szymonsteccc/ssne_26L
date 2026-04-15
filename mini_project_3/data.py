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

from nets import *


def load_data(data_folder, transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transform)
    return dataset


def split_dataset(dataset, split_ratio=0.8):
    total_size = len(dataset)
    size_1 = int(split_ratio * total_size)
    size_2 = total_size - size_1
    dataset_1, dataset_2 = random_split(dataset, [size_1, size_2])
    # dataset_1 = split_ratio * dataset
    return dataset_1, dataset_2

# def split_train_val_dataset(dataset, train_ratio=0.8):
#     total_size = len(dataset)
#     train_size = int(train_ratio * total_size)
#     val_size = total_size - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     return train_dataset, val_dataset

def train_test_split(X, y, test_size=0.2):
    total_size = len(X)
    test_size = int(test_size * total_size)
    indices = list(range(total_size))
    random.shuffle(indices)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

# nieuzywane
def number_of_classes(dataset):
    return len(dataset.classes)

def define_dataloaders(train_dataset, val_dataset, batch_size=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader