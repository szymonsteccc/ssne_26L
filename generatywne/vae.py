from numpy import concat
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_std = nn.Linear(hidden_dim, latent_dim)
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        mean_ = self.fc_mean(x)
        log_std_ = self.fc_log_std(x)
        return mean_, log_std_


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.view((-1, 3, 32, 32))
        return x


class VAE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int, output_dim: int, condition_size : int
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim + condition_size, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim + condition_size, hidden_dim, output_dim)

    def reparametrization_trick(self, mean_, log_std_):
        eps = torch.normal(torch.zeros_like(mean_), torch.ones_like(log_std_))
        z = mean_ + eps * torch.exp(0.5 * log_std_)
        return z

    def forward(self, x, class_emb):
        x = x.view(x.size(0), -1)
        mean_, log_std_ = self.encoder(torch.cat([x, class_emb], dim=1))
        z = self.reparametrization_trick(mean_, log_std_)
        x = self.decoder(torch.cat([z, class_emb], dim=1))
        return x, mean_, log_std_
