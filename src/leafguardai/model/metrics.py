from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    topk_accuracy: float
    macro_f1: float
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_index in range(num_classes):
        true_positive = np.sum((y_pred == class_index) & (y_true == class_index))
        false_positive = np.sum((y_pred == class_index) & (y_true != class_index))
        false_negative = np.sum((y_pred != class_index) & (y_true == class_index))

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(scores)) if scores else 0.0


def evaluate(model, loader, criterion, device, top_k=3, num_classes=None):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    topk_correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            k = min(top_k, outputs.size(1))
            topk_indices = outputs.topk(k, dim=1).indices
            topk_correct += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    if num_classes is None:
        num_classes = int(max(y_true_np.max(), y_pred_np.max()) + 1) if total else 0

    return EvalResult(
        loss=total_loss / total if total else 0.0,
        accuracy=correct / total if total else 0.0,
        topk_accuracy=topk_correct / total if total else 0.0,
        macro_f1=_macro_f1(y_true_np, y_pred_np, num_classes),
        y_true=y_true,
        y_pred=y_pred,
    )


def classification_report_text(y_true, y_pred, class_names) -> str:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lines = [
        f"{'classe':<45} {'prec':>8} {'rec':>8} {'f1':>8} {'suporte':>8}",
        "-" * 80,
    ]

    f1_scores = []
    for class_index, name in enumerate(class_names):
        true_positive = np.sum((y_pred == class_index) & (y_true == class_index))
        false_positive = np.sum((y_pred == class_index) & (y_true != class_index))
        false_negative = np.sum((y_pred != class_index) & (y_true == class_index))
        support = int(np.sum(y_true == class_index))

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        f1_scores.append(f1)
        lines.append(
            f"{name:<45} {precision:8.4f} {recall:8.4f} {f1:8.4f} {support:8d}"
        )

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    lines.append("-" * 80)
    lines.append(f"{'accuracy':<45} {accuracy:8.4f} {'':>8} {'':>8} {len(y_true):8d}")
    lines.append(f"{'macro avg (f1)':<45} {'':>8} {'':>8} {macro_f1:8.4f} {len(y_true):8d}")
    return "\n".join(lines)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[true_index, pred_index] += 1

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Matriz de confusão (conjunto de teste)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return matrix
