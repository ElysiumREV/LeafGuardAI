from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "color"


def ensure_dataset_exists():
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_PATH}. "
            "Baixe o PlantVillage (apenas as imagens coloridas) e coloque "
            "a pasta color/ nesse caminho. Veja o README."
        )


def get_classes():
    ensure_dataset_exists()
    return sorted(
        folder.name
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    )


def get_images_by_class():
    ensure_dataset_exists()
    dataset = []

    for folder in sorted(DATASET_PATH.iterdir()):
        if not folder.is_dir():
            continue

        images = [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for image in images:
            dataset.append(
                {
                    "image": image,
                    "label": folder.name,
                }
            )

    return dataset
