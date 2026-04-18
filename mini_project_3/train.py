import numpy as np
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


def train_model(net, train_loader, val_loader, criterion, optimizer, eval_func, device, num_epochs=5, verbose=True):
    train_eval_hist = []
    val_eval_hist = []
    loss_hist = []

    rng = tqdm(range(num_epochs)) if verbose else range(num_epochs)

    total = 0
    correct = 0

    best_val_acc = 0
    max_plateau = 4
    plateau_counter = 0

    for epoch in rng:
        net.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for data in train_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
              preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_hist.append(epoch_loss)

        train_acc = correct / total
        train_eval_hist.append(train_acc)
        val_eval_hist.append(eval_func(net, val_loader, device))

        if not verbose:
            print('[%d/%d] loss: %.3f' % (epoch + 1, num_epochs, epoch_loss))
        else:
            rng.set_postfix(
                loss=f"{epoch_loss:.3f}",
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_eval_hist[-1]:.3f}"
            )
            if (epoch+1) % 1 == 0:
              plot_training_chart(loss_hist, train_eval_hist, val_eval_hist)

        val_acc = val_eval_hist[-1]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            plateau_counter = 0
        else:
            plateau_counter += 1

        if plateau_counter >= max_plateau:
            print("Early stopping: validation accuracy stopped improving")
            break
    # train_eval_hist.append(eval_func(net, train_loader, device)) # Tutaj było 1 tab do przodu, czyli co epokę. TEraz szybeciej

    if not verbose:
        print('Finished Training')

    return loss_hist, train_eval_hist, val_eval_hist


def plot_training_chart(loss_hist, train_eval_hist, val_eval_hist):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(loss_hist, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)
    ax2.plot(train_eval_hist, label="train", color="blue")
    ax2.plot(val_eval_hist, label="val", color="cyan")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()
    plt.show()


def calc_accuracy(predictions: np.ndarray, targets: np.ndarray, n_classes=50):
    assert len(predictions) == len(targets)
    accuracies = []
    for i in range(n_classes):
        accuracies.append((predictions[targets == i] == i).sum() / (targets == i).sum())
    return np.mean(accuracies)


def get_accuracy(model, data, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
    return correct / total

