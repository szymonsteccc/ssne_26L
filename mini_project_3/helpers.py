import numpy as np


def calc_accuracy(predictions: np.ndarray, targets: np.ndarray, n_classes=50):
    assert len(predictions) == len(targets)
    accuracies = []
    for i in range(n_classes):
        accuracies.append((predictions[targets == i] == i).sum() / (targets == i).sum())
    return np.mean(accuracies)

import numpy as np
import pandas as pd


def load_predictions(csv_path):
    """
    Reads pred.csv -> returns filenames and predictions
    """
    data = pd.read_csv(csv_path, header=None)
    filenames = data[0].values
    preds = data[1].astype(int).values
    return filenames, preds


def create_synthetic_targets(preds, n_classes=50, match_ratio=0.5, seed=42):
    """
    Creates fake ground-truth targets:
    - 50% of the time: correct prediction
    - 50%: random wrong class
    """
    rng = np.random.default_rng(seed)

    targets = preds.copy()

    n = len(preds)
    idx = rng.choice(n, size=int(n * (1 - match_ratio)), replace=False)

    for i in idx:
        wrong_classes = list(range(n_classes))
        wrong_classes.remove(preds[i])
        targets[i] = rng.choice(wrong_classes)

    return targets


def evaluate_from_csv(csv_path, n_classes=50, match_ratio=0.5):
    """
    Full pipeline:
    - load pred.csv
    - create synthetic targets
    - compute accuracy using your function
    """

    filenames, preds = load_predictions(csv_path)
    targets = create_synthetic_targets(preds, n_classes, match_ratio)

    # reuse your metric
    acc = calc_accuracy(preds, targets, n_classes)

    print(f"Evaluated on {len(preds)} samples")
    print(f"Match ratio (synthetic): {match_ratio}")
    print(f"Mean class-balanced accuracy: {acc:.4f}")

    return acc, preds, targets

evaluate_from_csv("./mini_project_3/ZapisaneWyniki/pred_66%.csv")
