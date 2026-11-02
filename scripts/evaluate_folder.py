"""Avalia o modelo de models/ numa pasta de fotos organizada por classe.

A pasta deve ter uma subpasta por classe, com o nome exato da classe (ex.: Tomato___Late_blight/),
como data/teste_campo/. Usa o mesmo caminho de inferência do app (infer.predict).

Uso:
    uv run python scripts/evaluate_folder.py data/teste_campo
    uv run python scripts/evaluate_folder.py data/teste_campo --output resultado.csv

Gera uma planilha (CSV) com a previsão de cada foto e mostra um resumo: acerto geral,
acerto por faixa de confiança, por classe e as confusões mais comuns.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch

from leafguardai.model.infer import format_class_name, load_classes, load_model, predict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIDENCE_BANDS = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]


def main():
    parser = argparse.ArgumentParser(description="Avalia o modelo numa pasta de fotos organizada por classe.")
    parser.add_argument("folder", type=Path, help="Pasta com uma subpasta por classe")
    parser.add_argument("--output", type=Path, help="CSV de saída (padrão: <pasta>/resultado.csv)")
    args = parser.parse_args()

    output = args.output or args.folder / "resultado.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    model = load_model(len(classes), device)

    images = []
    for folder in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        if folder.name not in classes:
            print(f"aviso: '{folder.name}' não é uma classe do modelo; pasta ignorada")
            continue
        images += [(f, folder.name) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        print("Nenhuma imagem encontrada.")
        return

    rows = []
    for index, (path, label) in enumerate(images, start=1):
        top3 = predict(path, top_k=3, model=model, classes=classes, device=device)
        predicted, confidence = top3[0]
        rows.append({
            "arquivo": str(path.relative_to(args.folder)),
            "classe_correta": label,
            "previsto": predicted,
            "confianca": round(confidence, 4),
            "acertou": predicted == label,
            "correta_no_top3": label in [name for name, _ in top3],
            "top2": top3[1][0] if len(top3) > 1 else "",
            "top3": top3[2][0] if len(top3) > 2 else "",
        })
        if index % 250 == 0:
            print(f"  {index}/{len(images)}", flush=True)

    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(r["acertou"] for r in rows)
    in_top3 = sum(r["correta_no_top3"] for r in rows)
    print(f"\n{total} fotos | acerto: {correct / total * 100:.1f}% | correta entre as 3 primeiras: {in_top3 / total * 100:.1f}%")

    print("\nAcerto por faixa de confiança:")
    for low, high in CONFIDENCE_BANDS:
        band = [r for r in rows if low <= r["confianca"] < high]
        if band:
            rate = sum(r["acertou"] for r in band) / len(band) * 100
            print(f"  {int(low * 100):>3}-{min(100, int(high * 100))}%: {len(band):5d} fotos ({len(band) / total * 100:4.1f}%) | acerto {rate:5.1f}%")

    print("\nAcerto por classe (do pior para o melhor):")
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["classe_correta"]].append(r["acertou"])
    for label, hits in sorted(by_class.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {sum(hits) / len(hits) * 100:5.1f}%  ({len(hits):4d} fotos)  {format_class_name(label)}")

    print("\nConfusões mais comuns (correta -> prevista):")
    confusions = Counter((r["classe_correta"], r["previsto"]) for r in rows if not r["acertou"])
    for (label, predicted), count in confusions.most_common(10):
        print(f"  {count:4d}  {format_class_name(label)} -> {format_class_name(predicted)}")

    print(f"\nPlanilha salva em {output}")


if __name__ == "__main__":
    main()
