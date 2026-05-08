import torch
import torch.nn as nn

from helpers import corrupt


def train_diffusion(model, train_loader, device, epochs=5, lr=1e-3):
    model.train()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            noise_amount = torch.rand(x.shape[0], device=device)
            noisy_x = corrupt(x, noise_amount)

            pred = model(noisy_x, noise_amount)
            loss = loss_fn(pred, x)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        avg_loss = sum(losses[-len(train_loader) :]) / len(train_loader)
        print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

    return losses


def sample_diffusion(model, device, num_samples=64, steps=40, img_shape=(3, 32, 32)):
    model.eval()
    x = torch.rand(num_samples, *img_shape, device=device)
    with torch.no_grad():
        for i in range(steps):
            t = torch.zeros(x.size(0), device=device) + (steps - i) / steps
            pred = model(x, t)
            mix_factor = 1 / (steps - i)
            x = x * (1 - mix_factor) + pred * mix_factor
    return x
