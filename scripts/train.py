import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from pathlib import Path

from leafguardai.data.loader import get_classes, get_images_by_class
from leafguardai.data.dataset import PlantDataset
from leafguardai.data.split import split_dataset, create_dataloaders
from leafguardai.model.classifier import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def freeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = False


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Carregando dataset...", flush=True)
    samples = get_images_by_class()
    classes = get_classes()
    print(f"   -> {len(classes)} classes, {len(samples)} imagens", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = PlantDataset(samples=samples, classes=classes, transform=transform)

    print("2. Dividindo em treino/validação/teste...", flush=True)
    train_dataset, validation_dataset, test_dataset = split_dataset(dataset)
    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset, validation_dataset, test_dataset, batch_size=32
    )

    print("3. Criando modelo (features congeladas, só o classifier treina)...", flush=True)
    model = create_model(num_classes=len(classes))
    freeze_features(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.classifier.parameters(), lr=1e-3)

    epochs = 3
    best_val_acc = 0.0

    print("\n4. Treinando...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc = evaluate(model, validation_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint_path = MODELS_DIR / "best_model.pt"
            torch.save(model.state_dict(), checkpoint_path)

            # Salva a lista de classes na MESMA ordem usada pra treinar,
            # pra garantir que a inferência sempre use o mapeamento correto
            # índice -> nome da doença, mesmo que get_classes() retorne
            # uma ordem diferente no futuro.
            classes_path = MODELS_DIR / "classes.json"
            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(classes, f, ensure_ascii=False, indent=2)

            print(f"  -> novo melhor modelo salvo em {checkpoint_path}", flush=True)

    print("\n5. Avaliando no conjunto de teste...", flush=True)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f} | Test acc={test_acc:.4f}", flush=True)

    print("\n✅ Treinamento concluído!", flush=True)


if __name__ == "__main__":
    main()