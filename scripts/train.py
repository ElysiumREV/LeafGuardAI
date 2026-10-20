import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from leafguardai.data.dataset import PlantDataset
from leafguardai.data.loader import get_classes, get_images_by_class
from leafguardai.data.plantdoc import get_plantdoc_samples
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
from leafguardai.vision.background import BackgroundAugment
from leafguardai.vision.processor import ImageProcessor
from leafguardai.vision.transforms import (
    get_eval_transforms,
    get_field_train_transforms,
    get_train_transforms,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


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
    parser.add_argument(
        "--background-prob",
        type=float,
        default=0.0,
        help=(
            "Probabilidade de trocar o fundo de uma folha do PlantVillage (ex.: 0.5). Desligada por "
            "padrão: com o PlantDoc no treino não melhorou as fotos de campo e deixa a época ~3x mais lenta."
        ),
    )
    parser.add_argument(
        "--field-share",
        type=float,
        default=0.4,
        help="Fração de cada época vinda das fotos de campo (PlantDoc), se baixado.",
    )
    parser.add_argument(
        "--no-plantdoc",
        action="store_true",
        help="Não usa o PlantDoc mesmo que ele esteja em data/raw/PlantDoc.",
    )
    parser.add_argument(
        "--use-processor",
        action="store_true",
        help="Aplica o ImageProcessor no treino; a inferência passa a usá-lo automaticamente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODELS_DIR,
        help="Pasta dos artefatos (padrão: models/, que é a usada pelo app).",
    )
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


def save_checkpoint(model, classes, output_dir, use_processor):
    torch.save(model.state_dict(), output_dir / "best_model.pt")
    with open(output_dir / "classes.json", "w", encoding="utf-8") as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)
    # A inferência lê este arquivo para aplicar exatamente o pré-processamento do treino.
    with open(output_dir / "preprocessing.json", "w", encoding="utf-8") as file:
        json.dump({"use_processor": use_processor}, file, indent=2)


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
    field_validation_loader,
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
    best_score,
    output_dir,
    use_processor,
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
        message = (
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_acc={val_result.accuracy:.4f} | val_f1={val_result.macro_f1:.4f}"
        )

        # Com fotos de campo disponíveis, o checkpoint é escolhido por elas: é o que o app enfrenta.
        score = val_result.accuracy
        if field_validation_loader is not None:
            field_result = evaluate(
                model,
                field_validation_loader,
                criterion,
                device,
                top_k=top_k,
                num_classes=num_classes,
            )
            record["field_val_acc"] = field_result.accuracy
            record["field_val_top3"] = field_result.topk_accuracy
            message += f" | campo_val_acc={field_result.accuracy:.4f}"
            score = field_result.accuracy

        scheduler.step(score)
        history.append(record)
        print(message, flush=True)

        if score > best_score:
            best_score = score
            save_checkpoint(model, classes, output_dir, use_processor)
            print(f"  -> novo melhor modelo salvo em {output_dir / 'best_model.pt'}", flush=True)

        if early_stopping.step(score):
            print(
                f"  -> early stopping: sem melhora por {patience} épocas",
                flush=True,
            )
            break

    return best_score


def write_test_report(name, model, loader, criterion, device, classes, top_k, output_dir):
    result = evaluate(model, loader, criterion, device, top_k=top_k, num_classes=len(classes))
    print(
        f"[{name}] Test acc={result.accuracy:.4f} | Test top-{top_k}={result.topk_accuracy:.4f}",
        flush=True,
    )
    report = classification_report_text(result.y_true, result.y_pred, classes)
    (output_dir / f"classification_report{'' if name == 'plantvillage' else '_' + name}.txt").write_text(
        report + "\n", encoding="utf-8"
    )
    plot_confusion_matrix(
        result.y_true,
        result.y_pred,
        classes,
        output_dir / f"confusion_matrix{'' if name == 'plantvillage' else '_' + name}.png",
    )
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"
    output_dir = args.output_dir
    checkpoint_path = output_dir / "best_model.pt"

    print(f"Usando dispositivo: {device} | AMP: {use_amp}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Carregando dataset...", flush=True)
    samples = get_images_by_class()
    classes = get_classes()
    print(f"   -> {len(classes)} classes, {len(samples)} imagens", flush=True)

    if not samples:
        print("Nenhuma imagem encontrada. Verifique data/raw/PlantVillage/color/.", flush=True)
        return

    field_samples = [] if args.no_plantdoc else get_plantdoc_samples()
    if field_samples:
        print(f"   -> PlantDoc (fotos de campo): {len(field_samples)} imagens", flush=True)
    else:
        print(
            "   -> PlantDoc não encontrado: treinando só com PlantVillage. "
            "Fotos reais tendem a ter acurácia baixa (veja o README).",
            flush=True,
        )

    print("2. Split estratificado por folha treino/validação/teste...", flush=True)
    train_samples, validation_samples, test_samples = split_samples(
        samples,
        seed=args.seed,
    )
    field_train, field_validation, field_test = split_samples(
        field_samples,
        train_ratio=0.60,
        validation_ratio=0.15,
        test_ratio=0.25,
        seed=args.seed,
    )
    save_split(
        output_dir / "split.json",
        train_samples + field_train,
        validation_samples + field_validation,
        test_samples + field_test,
    )
    print(
        f"   -> treino={len(train_samples)} | val={len(validation_samples)} | teste={len(test_samples)}"
        + (f" | campo: treino={len(field_train)} val={len(field_validation)} teste={len(field_test)}"
           if field_samples else ""),
        flush=True,
    )

    processor = ImageProcessor() if args.use_processor else None
    background = None
    if args.background_prob > 0 and not args.use_processor:
        background = BackgroundAugment(args.background_prob, train_samples)
        masks = sum(1 for sample in train_samples if sample.get("mask"))
        print(f"   -> troca de fundo: p={args.background_prob} ({masks} imagens com máscara)", flush=True)

    train_dataset = PlantDataset(
        samples=train_samples + field_train,
        classes=classes,
        transform=get_train_transforms(),
        processor=processor,
        field_transform=get_field_train_transforms(),
        background=background,
    )
    validation_dataset = PlantDataset(
        samples=validation_samples,
        classes=classes,
        transform=get_eval_transforms(),
        processor=processor,
    )
    test_dataset = PlantDataset(
        samples=test_samples,
        classes=classes,
        transform=get_eval_transforms(),
        processor=processor,
    )

    # Sem reamostragem, as ~1.500 fotos de campo sumiriam entre dezenas de milhares do PlantVillage.
    train_sampler = None
    if field_train:
        n_lab, n_field = len(train_samples), len(field_train)
        weights = [(1 - args.field_share) / n_lab] * n_lab + [args.field_share / n_field] * n_field
        train_sampler = WeightedRandomSampler(weights, num_samples=n_lab + n_field, replacement=True)

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        train_sampler=train_sampler,
    )

    field_validation_loader = field_test_loader = None
    if field_samples:
        field_validation_loader, field_test_loader = (
            DataLoader(
                PlantDataset(part, classes, transform=get_eval_transforms(), processor=processor),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            for part in (field_validation, field_test)
        )

    print("3. Criando EfficientNet-B0...", flush=True)
    model = create_model(num_classes=len(classes))
    freeze_features(model)
    model = model.to(device)

    if args.no_class_weights:
        class_weights = None
    else:
        class_weights = compute_class_weights(train_samples + field_train, classes).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_score = 0.0
    common = dict(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        field_validation_loader=field_validation_loader,
        criterion=criterion,
        device=device,
        patience=args.patience,
        min_delta=args.min_delta,
        classes=classes,
        top_k=args.top_k,
        scaler=scaler,
        use_amp=use_amp,
        history=history,
        output_dir=output_dir,
        use_processor=args.use_processor,
    )

    print("\n4. Estágio 1 — backbone congelado, treina só o classifier...", flush=True)
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    best_score = run_stage(
        stage_name="stage1",
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs_stage1,
        best_score=best_score,
        **common,
    )

    print("\n5. Estágio 2 — descongela os últimos blocos do backbone...", flush=True)
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    best_score = run_stage(
        stage_name="stage2",
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs_stage2,
        best_score=best_score,
        **common,
    )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\n6. Avaliando o melhor checkpoint no conjunto de teste...", flush=True)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    test_result = write_test_report(
        "plantvillage", model, test_loader, criterion, device, classes, args.top_k, output_dir
    )
    metrics = {
        "best_val_score": best_score,
        "selected_by": "field_val_acc" if field_samples else "val_acc",
        "use_processor": args.use_processor,
        "test_loss": test_result.loss,
        "test_accuracy": test_result.accuracy,
        "test_macro_f1": test_result.macro_f1,
        f"test_top{args.top_k}": test_result.topk_accuracy,
    }
    if field_test_loader is not None:
        field_result = write_test_report(
            "campo", model, field_test_loader, criterion, device, classes, args.top_k, output_dir
        )
        # O PlantDoc cobre 28 das 38 classes; por isso não registramos macro-F1 sobre 38 classes.
        metrics["field_test_accuracy"] = field_result.accuracy
        metrics[f"field_test_top{args.top_k}"] = field_result.topk_accuracy
        metrics["field_test_size"] = len(field_test)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nArtefatos salvos em {output_dir}", flush=True)
    print("\nTreinamento concluído!", flush=True)


if __name__ == "__main__":
    main()
