import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from leafguardai.data.dataset import PlantDataset
from leafguardai.data.loader import get_classes, get_images_by_class
from leafguardai.data.split import create_dataloaders, save_split, split_samples
from leafguardai.model.classifier import (
    create_model,
    freeze_features,
    unfreeze_last_blocks,
)
from leafguardai.model.metrics import (
    classification_report_text,
    evaluate,
    plot_confusion_matrix,
)
from leafguardai.vision.transforms import get_eval_transforms, get_train_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value):
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina o classificador LeafGuardAI (EfficientNet-B0)."
    )
    parser.add_argument("--epochs-stage1", type=int, default=10)
    parser.add_argument("--epochs-stage2", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stage2-lr", type=float, default=1e-5)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def compute_class_weights(samples, classes):
    counts = Counter(sample["label"] for sample in samples)
    total = len(samples)
    num_classes = len(classes)
    weights = [
        total / (num_classes * counts.get(name, 1))
        for name in classes
    ]
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(model, classes):
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    classes_path = MODELS_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def run_stage(
    stage_name,
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    min_delta,
    classes,
    top_k,
    scaler,
    use_amp,
    history,
    best_val_acc,
):
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    num_classes = len(classes)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
        )
        val_result = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            top_k=top_k,
            num_classes=num_classes,
        )

        scheduler.step(val_result.accuracy)

        record = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_result.loss,
            "val_acc": val_result.accuracy,
            "val_f1": val_result.macro_f1,
            "val_top3": val_result.topk_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_result.loss:.4f} | val_acc={val_result.accuracy:.4f} | "
            f"val_f1={val_result.macro_f1:.4f} | val_top{top_k}={val_result.topk_accuracy:.4f}",
            flush=True,
        )

        if val_result.accuracy > best_val_acc:
            best_val_acc = val_result.accuracy
            save_checkpoint(model, classes)
            print(f"  -> novo melhor modelo salvo em {CHECKPOINT_PATH}", flush=True)

        if early_stopping.step(val_result.accuracy):
            print(
                f"  -> early stopping: val_acc sem melhora por {patience} épocas",
                flush=True,
            )
            break

    return best_val_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"

    print(f"Usando dispositivo: {device} | AMP: {use_amp}", flush=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Carregando dataset...", flush=True)
    samples = get_images_by_class()
    classes = get_classes()
    print(f"   -> {len(classes)} classes, {len(samples)} imagens", flush=True)

    if not samples:
        print("Nenhuma imagem encontrada. Verifique data/raw/PlantVillage/color/.", flush=True)
        return

    print("2. Split estratificado treino/validação/teste...", flush=True)
    train_samples, validation_samples, test_samples = split_samples(
        samples,
        seed=args.seed,
    )
    save_split(MODELS_DIR / "split.json", train_samples, validation_samples, test_samples)
    print(
        f"   -> treino={len(train_samples)} | "
        f"val={len(validation_samples)} | teste={len(test_samples)}",
        flush=True,
    )

    train_dataset = PlantDataset(
        samples=train_samples,
        classes=classes,
        transform=get_train_transforms(),
    )
    validation_dataset = PlantDataset(
        samples=validation_samples,
        classes=classes,
        transform=get_eval_transforms(),
    )
    test_dataset = PlantDataset(
        samples=test_samples,
        classes=classes,
        transform=get_eval_transforms(),
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("3. Criando EfficientNet-B0...", flush=True)
    model = create_model(num_classes=len(classes))
    freeze_features(model)
    model = model.to(device)

    if args.no_class_weights:
        class_weights = None
    else:
        class_weights = compute_class_weights(train_samples, classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_val_acc = 0.0

    print("\n4. Estágio 1 — backbone congelado, treina só o classifier...", flush=True)
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    best_val_acc = run_stage(
        stage_name="stage1",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs_stage1,
        patience=args.patience,
        min_delta=args.min_delta,
        classes=classes,
        top_k=args.top_k,
        scaler=scaler,
        use_amp=use_amp,
        history=history,
        best_val_acc=best_val_acc,
    )

    print("\n5. Estágio 2 — descongela os últimos blocos do backbone...", flush=True)
    if CHECKPOINT_PATH.exists():
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    unfreeze_last_blocks(model, n_blocks=args.unfreeze_blocks)
    optimizer = Adam(
        [
            {
                "params": [
                    param
                    for param in model.features.parameters()
                    if param.requires_grad
                ],
                "lr": args.stage2_lr,
            },
            {"params": model.classifier.parameters(), "lr": args.stage2_head_lr},
        ]
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    best_val_acc = run_stage(
        stage_name="stage2",
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs_stage2,
        patience=args.patience,
        min_delta=args.min_delta,
        classes=classes,
        top_k=args.top_k,
        scaler=scaler,
        use_amp=use_amp,
        history=history,
        best_val_acc=best_val_acc,
    )

    history_path = MODELS_DIR / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n6. Avaliando o melhor checkpoint no conjunto de teste...", flush=True)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    test_result = evaluate(
        model,
        test_loader,
        criterion,
        device,
        top_k=args.top_k,
        num_classes=len(classes),
    )
    print(
        f"Test loss={test_result.loss:.4f} | "
        f"Test acc={test_result.accuracy:.4f} | "
        f"Test F1={test_result.macro_f1:.4f} | "
        f"Test top-{args.top_k}={test_result.topk_accuracy:.4f}",
        flush=True,
    )

    report = classification_report_text(
        test_result.y_true,
        test_result.y_pred,
        classes,
    )
    report_path = MODELS_DIR / "classification_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)

    matrix_path = MODELS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(
        test_result.y_true,
        test_result.y_pred,
        classes,
        matrix_path,
    )

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val_acc,
                "test_loss": test_result.loss,
                "test_accuracy": test_result.accuracy,
                "test_macro_f1": test_result.macro_f1,
                f"test_top{args.top_k}": test_result.topk_accuracy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nRelatório salvo em {report_path}", flush=True)
    print(f"Matriz de confusão salva em {matrix_path}", flush=True)
    print(f"Histórico salvo em {history_path}", flush=True)
    print("\nTreinamento concluído!", flush=True)


if __name__ == "__main__":
    main()
