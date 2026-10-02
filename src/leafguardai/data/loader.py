from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = [
    PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "color",
    PROJECT_ROOT / "data" / "raw" / "NewPlantDataset" / "train",
    PROJECT_ROOT / "data" / "raw" / "NewPlantDataset" / "train2"
]


def ensure_dataset_exists():
    for path in DATASET_PATH:
        if not path.is_dir():
            raise FileNotFoundError(
                f"Dataset não encontrado em {path}. "
                "Verifique se as pastas dos datasets estão nos caminhos corretos. Veja o README."
            )


def get_classes():
    ensure_dataset_exists()
    classes = set()
    for path in DATASET_PATH:
        for folder in path.iterdir():
            if folder.is_dir():
                classes.add(folder.name)
    return sorted(list(classes))


def get_images_by_class():
    ensure_dataset_exists()
    dataset = []

    for path in DATASET_PATH:
        for folder in sorted(path.iterdir()):
            if not folder.is_dir():
                continue

            # Filtro: Ignorar pastas genéricas que não seguem o padrão "Planta___Doenca"
            # Isso evita que pastas como "Disease", "Healthy" ou "dataset" entrem como classes.
            folder_name = folder.name
            if "___" not in folder_name:
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
                        "label": folder_name,
                    }
                )

    return dataset
