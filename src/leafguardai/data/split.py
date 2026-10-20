import json
import random
import re
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader

# Sufixos das cópias aumentadas do NewPlantDataset e das cópias " copy N" do PlantVillage.
_COPY_SUFFIX = re.compile(
    r"(_flipTB|_flipLR|_90deg|_180deg|_270deg|_new30degFlipLR|_change_\d+|_final_masked| copy(?: \d+)?)+$",
    re.IGNORECASE,
)


def leaf_group(label, filename):
    """Identificador da foto original: cópias, flips e rotações da mesma folha têm o mesmo grupo.

    Ex.: 'uuid___RS_LB 4640.JPG', 'uuid2___RS_LB 4640 copy.JPG' e 'uuid___RS_LB 4640_flipTB.JPG'
    -> 'Potato___Late_blight:rs_lb 4640'.
    """
    stem = filename.rsplit(".", 1)[0]
    photo = stem.split("___", 1)[1] if "___" in stem else stem
    return f"{label}:{_COPY_SUFFIX.sub('', photo).strip().lower()}"


def split_samples(
    samples,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """Split estratificado por classe e agrupado por foto: amostras com o mesmo "group"
    ficam sempre na mesma parte (evita a mesma folha no treino e no teste)."""
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("As proporções precisam somar 1.0")

    rng = random.Random(seed)
    by_class = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        group = sample.get("group") or str(sample["image"])
        by_class[sample["label"]][group].append(sample)

    train_samples = []
    validation_samples = []
    test_samples = []

    for label in sorted(by_class):
        groups = [by_class[label][key] for key in sorted(by_class[label])]
        rng.shuffle(groups)
        total = len(groups)
        train_size = int(total * train_ratio)
        validation_size = int(total * validation_ratio)

        for items in groups[:train_size]:
            train_samples.extend(items)
        for items in groups[train_size:train_size + validation_size]:
            validation_samples.extend(items)
        for items in groups[train_size + validation_size:]:
            test_samples.extend(items)

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
    train_sampler=None,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
