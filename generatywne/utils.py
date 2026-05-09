import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg

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
    classes : torch.tensor,
    num_imgs: int = 1000,
    single_img_shape: tuple = (3, 32, 32),
    device: str = "cuda",
) -> torch.tensor:

    z = torch.randn([num_imgs, model.latent_dim]).to(device)
    class_emb = onehot[classes].to(device)
    dec_input = torch.cat([z, class_emb], dim=1)
    
    imgs = model.decoder(dec_input.to(device))
    assert imgs.shape == tuple((num_imgs, *single_img_shape))
    return imgs


def save_images(imgs: torch.tensor, filename: str) -> None:
    torch.save(imgs.cpu().detach(), filename)


def save_model(model: VAE, filename: str) -> None:
    torch.save(model.state_dict(), filename)

def get_features(loader, model, device):
    features = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if x.shape[1] != 3:  # grayscale -> 3 channels
                x = x.repeat(1, 3 // x.shape[1], 1, 1)
            # resize to 299x299 (InceptionV3 default)
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            feat = model(x)
            features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features
######### Frechet Inception distance based on implementation from https://github.com/mseitzer/pytorch-fid

def calculate_frechet_distance(distribution_1, distribution_2, eps=1e-6):
    mu1 = np.mean(distribution_1, axis=0)
    sigma1 = np.cov(distribution_1, rowvar=False)

    mu2 = np.mean(distribution_2, axis=0)
    sigma2 = np.cov(distribution_2, rowvar=False)

    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)