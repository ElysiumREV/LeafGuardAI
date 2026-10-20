"""Baixa o PlantDoc (fotos de campo, CC BY 4.0) para data/raw/PlantDoc.

Fonte: https://github.com/pratikkayal/PlantDoc-Dataset
Alguns nomes de arquivo do repositório têm caracteres inválidos no Windows (ex.: "?");
eles são trocados por "_" na extração.
"""
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from leafguardai.data.plantdoc import PLANTDOC_PATH

ZIP_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
INVALID_CHARS = re.compile(r'[<>:"|?*]')


def _progress(blocks, block_size, total):
    done = blocks * block_size
    if blocks % 6400 == 0:  # a cada ~50 MB
        print(f"   baixado: {done / 1024**2:,.0f} MB", flush=True)


def main():
    if PLANTDOC_PATH.is_dir() and any(PLANTDOC_PATH.iterdir()):
        print(f"PlantDoc já existe em {PLANTDOC_PATH}. Apague a pasta para baixar de novo.")
        return

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "plantdoc.zip"
        print(f"1. Baixando {ZIP_URL} (~1 GB)...")
        urllib.request.urlretrieve(ZIP_URL, zip_path, _progress)
        print()

        print(f"2. Extraindo para {PLANTDOC_PATH}...")
        count = 0
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                parts = member.filename.split("/")[1:]  # remove "PlantDoc-Dataset-master/"
                if member.is_dir() or len(parts) < 3 or parts[0] not in {"train", "test"}:
                    continue
                target = PLANTDOC_PATH.joinpath(*(INVALID_CHARS.sub("_", part) for part in parts))
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, open(target, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                count += 1

    print(f"Pronto: {count} imagens em {PLANTDOC_PATH}")
    print("O treino (scripts/train.py) passa a usá-las automaticamente.")


if __name__ == "__main__":
    main()
