from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from leafguardai.data.loader import (
    get_classes,
    get_images_by_class
)

from leafguardai.data.dataset import PlantDataset


samples = get_images_by_class()
classes = get_classes()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


dataset = PlantDataset(
    samples=samples,
    classes=classes,
    transform=transform
)

total_size = len(dataset)

train_size = int(total_size * 0.70)
validation_size = int(total_size * 0.15)
test_size = total_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(
    dataset,
    [train_size, validation_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

image, label = dataset[0]

print("Total:", len(dataset))
print("Train:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Test:", len(test_dataset))

images, labels = next(iter(train_loader))

print("Images:", images.shape)
print("Labels:", labels.shape)
