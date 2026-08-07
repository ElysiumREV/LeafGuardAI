from leafguardai.data.loader import get_images_by_class, get_classes
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

dataset = get_images_by_class()
classes = get_classes()
sample = dataset[0]

print(
    "Total de imagens:",
    len(dataset)
)

print(
    dataset[0]
)

print(
    "Total de classes:",
    len(classes)
)

for c in classes:
    print(c)

labels = [
  item["label"]
  for item in dataset
]

counter = Counter(labels)

for label, amount in counter.items():
  print(label, "->", amount)

image = Image.open(
  sample["image"]
)

plt.imshow(image)
plt.title(sample["label"])
plt.axis("off")
plt.show()
