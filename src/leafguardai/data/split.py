import json
import random
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader


def split_samples(
    samples,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("As proporções precisam somar 1.0")

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample["label"]].append(sample)

    train_samples = []
    validation_samples = []
    test_samples = []

    for items in by_class.values():
        rng.shuffle(items)
        total = len(items)
        train_size = int(total * train_ratio)
        validation_size = int(total * validation_ratio)

        train_samples.extend(items[:train_size])
        validation_samples.extend(items[train_size:train_size + validation_size])
        test_samples.extend(items[train_size + validation_size:])

    rng.shuffle(train_samples)
    rng.shuffle(validation_samples)
    rng.shuffle(test_samples)

    return train_samples, validation_samples, test_samples


def save_split(path: Path, train_samples, validation_samples, test_samples):
    payload = {
        "train": [str(sample["image"]) for sample in train_samples],
        "validation": [str(sample["image"]) for sample in validation_samples],
        "test": [str(sample["image"]) for sample in test_samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, validation_loader, test_loader
