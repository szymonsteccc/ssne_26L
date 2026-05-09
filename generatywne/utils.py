import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from vae import VAE


def corrupt(x, amount):
    """Corrupt the input x by mixing it with noise according to amount."""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

def vae_loss(
    x_out: torch.tensor, x: torch.tensor, mean_: torch.tensor, log_std_: torch.tensor
) -> torch.tensor:
    mse = nn.functional.mse_loss(x_out, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_std_ - mean_.pow(2) - log_std_.exp())
    return mse + kld


def train(
    model: VAE,
    optimizer: Optimizer,
    loader: DataLoader,
    losses_list: list,
    onehot : nn.functional.one_hot,
    criterion: callable = vae_loss,
    overfit_on_single_batch: bool = False,
    device: str = "cuda",
    epochs: int = 10,
    show_every: int = 1,
):
    for epoch in range(1, epochs + 1):
        losses_epoch = []
        for x, y in loader:
            # print(y)
            x = x.to(device)
            optimizer.zero_grad()
            x_out, mean_, log_std_ = model(x, onehot[y].to(device))
            loss = criterion(x_out, x, mean_, log_std_)
            if not torch.isnan(loss):
                loss.backward()
                losses_epoch.append(loss.item())
                optimizer.step()
            if overfit_on_single_batch:
                break
        losses_list += losses_epoch
        if epoch and not epoch % show_every:
            print(f"[{epoch}/{epochs}], Loss: {torch.mean(torch.tensor(losses_epoch)).item():.3}")


def generate_random_images(
    model: VAE,
    onehot : nn.functional.one_hot,
    classes : list,
    num_imgs: int = 1000,
    single_img_shape: tuple = (3, 32, 32),
    device: str = "cuda",
) -> torch.tensor:

    z = torch.randn([num_imgs, model.latent_dim]).to(device)
    class_emb = onehot[classes].float().to(device)
    dec_input = torch.cat([z, class_emb], dim=1)
    
    imgs = model.decoder(dec_input.to(device))
    assert imgs.shape == tuple((num_imgs, *single_img_shape))
    return imgs


def save_images(imgs: torch.tensor, filename: str) -> None:
    torch.save(imgs.cpu().detach(), filename)


def save_model(model: VAE, filename: str) -> None:
    torch.save(model.state_dict(), filename)