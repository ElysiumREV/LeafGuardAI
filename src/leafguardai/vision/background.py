"""Troca de fundo das folhas do PlantVillage usando as máscaras de PlantVillage/segmented.

O PlantVillage tem quase sempre o mesmo fundo de laboratório; sem variar o fundo, o modelo
erra muito quando a folha aparece sobre outro fundo (ex.: 99% -> 36% num fundo cinza liso).
"""
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEGMENTED_PATH = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "segmented"

_masks_by_class = {}


def find_mask_path(label, filename):
    """Caminho da máscara em PlantVillage/segmented para uma imagem de PlantVillage/color (ou None)."""
    if label not in _masks_by_class:
        folder = SEGMENTED_PATH / label
        index = {}
        if folder.is_dir():
            for file in folder.iterdir():
                stem = file.name[: -len("_final_masked.jpg")] if file.name.endswith("_final_masked.jpg") else None
                if stem is None:
                    continue
                index[stem.lower()] = file
                # Algumas classes (ex.: Corn Common rust) têm UUID só no nome da máscara.
                if "___" in stem:
                    index.setdefault(stem.split("___", 1)[1].lower(), file)
        _masks_by_class[label] = index
    stem = Path(filename).stem.lower()
    return _masks_by_class[label].get(stem)


def load_mask(mask_path, size):
    """Máscara binária (uint8 0/255) da folha, preenchida, ou None se não for utilizável."""
    segmented = cv2.imread(str(mask_path))
    if segmented is None or (segmented.shape[1], segmented.shape[0]) != tuple(size):
        return None
    mask = (segmented.max(axis=2) > 25).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count > 1:
        mask = ((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled if filled.mean() >= 5 else None


def _random_color(rng):
    hsv = np.uint8([[[rng.randint(0, 179), rng.randint(0, 255), rng.randint(30, 255)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def make_background(rng, height, width, leaf_pool):
    """Fundo aleatório: cor lisa, degradê, textura ou folhagem desfocada feita de outras folhas."""
    noise = np.random.default_rng(rng.randint(0, 2**31))
    kind = rng.choice(["solid", "gradient", "texture", "foliage", "foliage"])
    if kind == "solid" or (kind == "foliage" and not leaf_pool):
        background = np.ones((height, width, 3), np.float32) * _random_color(rng)
    elif kind == "gradient":
        start, end = _random_color(rng), _random_color(rng)
        t = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        background = np.broadcast_to(start * (1 - t) + end * t, (height, width, 3)).copy()
        if rng.random() < 0.5 and height == width:
            background = np.ascontiguousarray(background.transpose(1, 0, 2))
    elif kind == "texture":
        cells = rng.choice([4, 8, 16, 32])
        small = noise.random((cells, cells, 3)).astype(np.float32)
        background = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC) * rng.uniform(60, 200)
        background = background + _random_color(rng) * rng.uniform(0.3, 0.8)
    else:
        background = np.ones((height, width, 3), np.float32) * (_random_color(rng) * 0.5 + np.float32([40, 70, 30]) * 0.5)
        for _ in range(rng.randint(5, 12)):
            leaf, mask = leaf_pool[rng.randrange(len(leaf_pool))]
            matrix = cv2.getRotationMatrix2D((leaf.shape[1] / 2, leaf.shape[0] / 2), rng.uniform(0, 360), rng.uniform(0.4, 1.2))
            matrix[:, 2] += [rng.uniform(-width / 2, width / 2), rng.uniform(-height / 2, height / 2)]
            warped = cv2.warpAffine(leaf, matrix, (width, height)).astype(np.float32)
            alpha = cv2.warpAffine(mask, matrix, (width, height)).astype(np.float32)[..., None] / 255
            background = warped * alpha + background * (1 - alpha)
        background = cv2.GaussianBlur(background, (0, 0), rng.uniform(1.5, 6))
    background = background + noise.normal(0, rng.uniform(0, 8), background.shape)
    return np.clip(background, 0, 255).astype(np.float32)


def composite(rgb, mask, background, rng=None, rescale_probability=0.0):
    """Cola a folha (rgb + máscara) sobre o fundo; opcionalmente diminui e desloca a folha."""
    height, width = mask.shape
    if rng is not None and rng.random() < rescale_probability:
        scale = rng.uniform(0.55, 1.0)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
        matrix[:, 2] += [rng.uniform(-(1 - scale) * width / 2, (1 - scale) * width / 2),
                         rng.uniform(-(1 - scale) * height / 2, (1 - scale) * height / 2)]
        rgb = cv2.warpAffine(rgb, matrix, (width, height))
        mask = cv2.warpAffine(mask, matrix, (width, height))
    alpha = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255
    return (rgb.astype(np.float32) * alpha + background * (1 - alpha)).clip(0, 255).astype(np.uint8)


class BackgroundAugment:
    """Com probabilidade `probability`, coloca a folha sobre um fundo aleatório (só no treino)."""

    def __init__(self, probability, pool_samples, pool_size=150):
        self.probability = probability
        self.pool_samples = [s for s in pool_samples if s.get("mask")]
        self.pool_size = pool_size
        self._pool = None

    def _leaf_pool(self):
        # Montado sob demanda dentro de cada worker do DataLoader.
        if self._pool is None:
            self._pool = []
            candidates = random.sample(self.pool_samples, min(len(self.pool_samples), self.pool_size * 2))
            for sample in candidates:
                with Image.open(sample["image"]) as file:
                    rgb = np.array(file.convert("RGB"))
                mask = load_mask(sample["mask"], (rgb.shape[1], rgb.shape[0]))
                if mask is not None:
                    self._pool.append((rgb, mask))
                if len(self._pool) >= self.pool_size:
                    break
        return self._pool

    def __call__(self, image, mask_path):
        if random.random() >= self.probability:
            return image
        mask = load_mask(mask_path, image.size)
        if mask is None:
            return image
        background = make_background(random, mask.shape[0], mask.shape[1], self._leaf_pool())
        return Image.fromarray(composite(np.array(image), mask, background, random, rescale_probability=0.5))
