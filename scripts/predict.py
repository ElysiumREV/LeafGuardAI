import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from leafguardai.model.classifier import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
CLASSES_PATH = MODELS_DIR / "classes.json"


def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CLASSES_PATH}. Rode a versão atualizada de "
            "scripts/train.py primeiro (ela salva classes.json)."
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(num_classes, device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CHECKPOINT_PATH}. Rode scripts/train.py primeiro."
        )

    model = create_model(num_classes=num_classes)

    # map_location garante que o checkpoint carrega mesmo que tenha sido
    # treinado num dispositivo diferente do que está rodando agora
    # (ex: treinou em GPU, está usando em CPU).
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def preprocess_image(image_path):
    # Mesmo pré-processamento usado no treino (train.py). Precisa ser
    # idêntico, senão o modelo recebe imagens numa distribuição diferente
    # da que ele aprendeu.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image)

    return tensor.unsqueeze(0)  # adiciona a dimensão de batch: (3,224,224) -> (1,3,224,224)


def predict(image_path, top_k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = load_classes()
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


def main():
    parser = argparse.ArgumentParser(
        description="Classifica uma imagem de folha com o modelo treinado do LeafGuardAI"
    )
    parser.add_argument("image", type=str, help="Caminho da imagem a ser analisada")
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Quantas classes mais prováveis mostrar (padrão: 3)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        return

    results = predict(image_path, top_k=args.top)

    print(f"\nResultado para: {image_path.name}\n")
    for rank, (label, prob) in enumerate(results, start=1):
        print(f"{rank}. {label:<40} {prob * 100:.2f}%")


if __name__ == "__main__":
    main()