import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_model(num_classes):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def freeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model, n_blocks=3):
    freeze_features(model)
    blocks = list(model.features.children())[-n_blocks:]
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True


def trainable_parameters(model):
    return [param for param in model.parameters() if param.requires_grad]
