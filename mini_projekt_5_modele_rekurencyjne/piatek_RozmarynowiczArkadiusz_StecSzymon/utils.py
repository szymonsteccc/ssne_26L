import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: str = "cuda",
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, lengths, targets in loader:
            x = x.to(device)
            targets = targets.to(device)

            current_batch_size = x.size(0)
            hidden_state = model.init_hidden(current_batch_size, device)

            preds, _ = model(x, hidden_state)

            loss = criterion(preds, targets)

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

    return total_loss / total_samples


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda"
) -> float:

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

    accuracy = 100 * correct_predictions / total_samples
    return accuracy


def train_composer_classifier(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    losses_list: list,
    criterion=nn.CrossEntropyLoss(),
    device: str = "cuda",
    epochs: int = 10,
    show_every: int = 1,
    patience: int = 5,
    save_path: str = "best_model.pt",
):
    best_valid_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):

        # ---------------- TRAIN ----------------
        model.train()

        losses_epoch = []

        for x, lengths, targets in train_loader:

            x = x.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            current_batch_size = x.size(0)
            hidden_state = model.init_hidden(current_batch_size, device)

            preds, _ = model(x, hidden_state)

            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            losses_epoch.append(loss.item())

        train_loss = sum(losses_epoch) / len(losses_epoch)

        # ---------------- VALIDATION ----------------
        valid_loss = evaluate_loss(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        valid_acc = evaluate_accuracy(
            model=model,
            loader=valid_loader,
            device=device,
        )

        losses_list.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
        })

        # ---------------- SAVE BEST MODEL ----------------
        if valid_loss < best_valid_loss:

            best_valid_loss = valid_loss
            epochs_without_improvement = 0

            torch.save(model.state_dict(), save_path)

            print(f"Saved best model -> {save_path}")

        else:
            epochs_without_improvement += 1

        # ---------------- LOGGING ----------------
        if epoch % show_every == 0:
            print(
                f"[{epoch}/{epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"Valid Acc: {valid_acc:.2f}%"
            )

        # ---------------- OVERFITTING / EARLY STOPPING ----------------
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Best validation loss: {best_valid_loss:.4f}")


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


def test_pad_collate(batch):

    lengths = torch.tensor([len(seq) for seq in batch])

    sequences = [
        torch.tensor(seq, dtype=torch.long) + 1
        for seq in batch
    ]

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    return padded_sequences, lengths
