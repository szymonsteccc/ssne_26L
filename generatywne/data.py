import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

DEFAULT_MEAN = (0.3184, 0.2927, 0.3013)
DEFAULT_STD = (0.2763, 0.2655, 0.2686)


def build_transforms(mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def load_data(
    root="trafic_32",
    batch_size=32,
    split=0.8,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    num_workers=0,
    seed=42,
):
    dataset = ImageFolder(root, transform=build_transforms(mean, std))
    train_size = int(split * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return dataset, train_dataset, test_dataset, train_loader, test_loader
