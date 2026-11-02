import time

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from typing import Optional

from leafguardai.vision.background import BackgroundAugment
from leafguardai.vision.processor import ImageProcessor

class PlantDataset(Dataset):

    def __init__(
        self,
        samples,
        classes,
        transform=None,
        processor: Optional[ImageProcessor] = None,
        field_transform=None,
        background: Optional[BackgroundAugment] = None,
    ):
        self.samples = samples
        self.classes = classes
        self.transform = transform
        self.processor = processor
        # Transform alternativo para fotos de campo, que têm enquadramentos variados.
        self.field_transform = field_transform
        # Troca de fundo (só para amostras do PlantVillage que têm máscara).
        self.background = background

        self.class_to_index = {
            name: index
            for index, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _open(path, attempts=3):
        # Em HDs lentos, uma leitura sob muitos acessos simultâneos pode falhar uma vez
        # ("broken data stream") com o arquivo íntegro; tentar de novo evita derrubar o treino.
        for attempt in range(attempts):
            try:
                with Image.open(path) as file:
                    return ImageOps.exif_transpose(file).convert("RGB")
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(0.5)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.processor:
            image = self.processor.process(sample["image"])
        else:
            image = self._open(sample["image"])
            if self.background is not None and sample.get("mask"):
                image = self.background(image, sample["mask"])

        transform = self.transform
        if self.field_transform is not None and sample.get("source") == "field":
            transform = self.field_transform
        if transform:
            image = transform(image)

        label = self.class_to_index[sample["label"]]

        return image, label
