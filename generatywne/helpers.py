import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corrupt(x, amount):
    """Corrupt the input x by mixing it with noise according to amount."""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount


def denormalize(x, mean, std):
    mean_t = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device).view(1, -1, 1, 1)
    out = x * std_t + mean_t
    return out.clamp(0.0, 1.0)


def save_samples(samples, path):
    torch.save(samples.cpu().detach(), path)
