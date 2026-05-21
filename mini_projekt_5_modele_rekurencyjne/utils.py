import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_composer_classifier(
    model: nn.Module,
    optimizer: Optimizer,
    loader: DataLoader,
    losses_list: list,
    criterion: callable = nn.CrossEntropyLoss(),
    device: str = "cuda",
    epochs: int = 10,
    show_every: int = 1,
):
    model.train()
    
    for epoch in range(1, epochs + 1):
        losses_epoch = []
        
        for x, lengths, targets in loader:
            x = (x).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            current_batch_size = x.size(0)
            hidden_state = model.init_hidden(current_batch_size, device)
            preds, _ = model(x, hidden_state)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())
                            
        losses_list += losses_epoch        
        if epoch and not epoch % show_every:
            epoch_mean_loss = torch.mean(torch.tensor(losses_epoch)).item() if losses_epoch else 0.0
            print(f"[{epoch}/{epochs}], Loss: {epoch_mean_loss:.4f}")


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str = "cuda") -> float:
    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, lengths, targets in loader:
            x = x.to(device)
            targets = targets.to(device)
            
            current_batch_size = x.size(0)
            hidden_state = model.init_hidden(current_batch_size, device)
            preds, _ = model(x, hidden_state)
            
            predictions = torch.argmax(preds, dim=1)
            
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy


def pad_collate(batch):
    sequences, labels = zip(*batch)

    # oryginalne długości i zmaina na tensory
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences = [torch.tensor(seq, dtype=torch.long) + 1 for seq in sequences]
    
    # padding
    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )
    labels = torch.tensor(labels)
    
    return padded_sequences, lengths, labels

