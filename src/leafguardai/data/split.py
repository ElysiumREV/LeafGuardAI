import torch
from torch.utils.data import random_split, DataLoader


def split_dataset(
    dataset,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    if train_ratio + validation_ratio + test_ratio != 1.0:
        raise ValueError("As proporções precisam somar 1.0")

    total_size = len(dataset)

    train_size = int(total_size * train_ratio)
    validation_size = int(total_size * validation_ratio)
    test_size = total_size - train_size - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        [train_size, validation_size, test_size],
        generator=generator,
    )

    return train_dataset, validation_dataset, test_dataset


def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=32,
    num_workers=0,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader, test_loader