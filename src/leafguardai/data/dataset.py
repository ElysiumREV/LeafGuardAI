from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from leafguardai.vision.processor import ImageProcessor

class PlantDataset(Dataset):

    def __init__(self, samples, classes, transform=None, processor: Optional[ImageProcessor] = None):
        self.samples = samples
        self.classes = classes
        self.transform = transform
        self.processor = processor

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
            image = Image.open(sample["image"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_index[sample["label"]]

        return image, label
