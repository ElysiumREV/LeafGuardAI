import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from leafguardai.model.classifier import create_model
from leafguardai.vision.transforms import get_eval_transforms
from leafguardai.vision.processor import ImageProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
CLASSES_PATH = MODELS_DIR / "classes.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"


def format_class_name(name: str) -> str:
    """
    Formats class names from 'Planta___Doenca' to a human-readable format.
    Example: 'Apple___Apple_scab' -> 'Apple Scab'
    """
    if "___" not in name:
        return name.replace("_", " ").capitalize()

    # Split into plant and disease
    plant, disease = name.split("___", 1)

    # Clean disease name: replace underscores with spaces
    disease_clean = disease.replace("_", " ")

    # Avoid redundancy: If disease name starts with plant name, remove it
    # Example: 'Apple_scab' -> 'Scab' when plant is 'Apple'
    if disease_clean.lower().startswith(plant.lower()):
        # Remove plant name and any trailing underscores/spaces
        disease_clean = disease_clean[len(plant):].strip(" _-")

    # Capitalize words
    final_name = f"{plant} {disease_clean}".strip()
    return " ".join(word.capitalize() for word in final_name.split())


def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CLASSES_PATH}. Rode scripts/train.py primeiro "
            "(o treino salva classes.json junto com o checkpoint)."
        )

    with open(CLASSES_PATH, encoding="utf-8") as file:
        return json.load(file)


def load_model(num_classes, device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CHECKPOINT_PATH}. Rode scripts/train.py primeiro."
        )

    model = create_model(num_classes=num_classes)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def uses_processor():
    """Se a inferência deve aplicar o ImageProcessor.

    scripts/train.py grava models/preprocessing.json com o pré-processamento usado no
    treino, e a inferência repete exatamente o mesmo (usar um diferente derruba a acurácia:
    99% -> 64% em folhas do PlantVillage). Checkpoints antigos, sem esse arquivo, mantêm o
    comportamento original do app (com processor).
    """
    if not PREPROCESSING_PATH.exists():
        return True
    with open(PREPROCESSING_PATH, encoding="utf-8") as file:
        return bool(json.load(file).get("use_processor", False))


def preprocess_image(image_path, use_processor=None):
    if use_processor is None:
        use_processor = uses_processor()
    if use_processor:
        image = ImageProcessor().process(image_path)
    else:
        with Image.open(image_path) as file:
            image = ImageOps.exif_transpose(file).convert("RGB")

    transform = get_eval_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict(image_path, top_k=3, model=None, classes=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if classes is None:
        classes = load_classes()

    if model is None:
        model = load_model(num_classes=len(classes), device=device)

    input_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_k = min(top_k, len(classes))
    top_probs, top_indices = probabilities.topk(top_k)

    return [
        (classes[index.item()], prob.item())
        for prob, index in zip(top_probs, top_indices)
    ]
