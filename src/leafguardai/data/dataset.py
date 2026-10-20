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
        # Transform alternativo para fotos de campo (PlantDoc), que têm enquadramentos variados.
        self.field_transform = field_transform
        # Troca de fundo (só para amostras do PlantVillage que têm máscara).
        self.background = background

        self.class_to_index = {
            name: index
            for index, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.processor:
            image = self.processor.process(sample["image"])
        else:
            with Image.open(sample["image"]) as file:
                image = ImageOps.exif_transpose(file).convert("RGB")
            if self.background is not None and sample.get("mask"):
                image = self.background(image, sample["mask"])

        transform = self.transform
        if self.field_transform is not None and sample.get("source") == "plantdoc":
            transform = self.field_transform
        if transform:
            image = transform(image)

        label = self.class_to_index[sample["label"]]

        return image, label
