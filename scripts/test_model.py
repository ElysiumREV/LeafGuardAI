from leafguardai.data.loader import get_classes
from leafguardai.model.classifier import create_model


classes = get_classes()

model = create_model(
    num_classes=len(classes)
)

print(model)
