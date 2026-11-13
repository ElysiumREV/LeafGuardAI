"""Empacota o modelo treinado para publicar no GitHub Release.

Uso:
    uv run python scripts/package_model.py
    uv run python scripts/package_model.py --models-dir models_novo

Gera dist/leafguardai-model.zip com best_model.pt, classes.json e preprocessing.json
(e metrics.json, se existir) e dist/leafguardai-model.zip.sha256 com o hash do zip.
Publique OS DOIS arquivos como assets de um release no GitHub, com exatamente estes nomes
(o app confere o hash antes de instalar o modelo): o app baixa o modelo de
https://github.com/ElysiumREV/LeafGuardAI/releases/latest/download/leafguardai-model.zip
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("best_model.pt", "classes.json", "preprocessing.json")
OPTIONAL = ("metrics.json",)


def main():
    parser = argparse.ArgumentParser(description="Empacota o modelo para o GitHub Release.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "dist" / "leafguardai-model.zip")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (args.models_dir / name).exists()]
    if missing:
        raise SystemExit(f"Faltam arquivos em {args.models_dir}: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as package:
        for name in REQUIRED + OPTIONAL:
            if (args.models_dir / name).exists():
                package.write(args.models_dir / name, name)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    checksum_path = args.output.with_name(args.output.name + ".sha256")
    checksum_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")

    print(f"Pacote criado: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA-256: {checksum_path} (publique os DOIS arquivos no release)")
    metrics_path = args.models_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "field_test_accuracy" in metrics:
            print(f"Acerto em fotos de campo: {metrics['field_test_accuracy'] * 100:.1f}% "
                  f"(top-3 {metrics['field_test_top3'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
