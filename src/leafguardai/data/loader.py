from pathlib import Path

from leafguardai.data.split import leaf_group
from leafguardai.vision.background import find_mask_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRIMARY_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "color"

# NewPlantDataset é o próprio PlantVillage com cópias aumentadas (flip/rotação); o split por
# folha (split.leaf_group) mantém essas cópias junto da original. "train2" era uma cópia
# idêntica de "train" e foi removida para não duplicar imagens.
DATASET_PATH = [
    PRIMARY_DATASET_PATH,
    PROJECT_ROOT / "data" / "raw" / "NewPlantDataset" / "train",
]


def ensure_dataset_exists():
        if not PRIMARY_DATASET_PATH.is_dir():
            raise FileNotFoundError(
                f"Dataset principal não encontrado em {PRIMARY_DATASET_PATH}. "
                "Verifique se o PlantVillage está no caminho correto. Veja o README."
            )

def get_available_dataset_paths():
    ensure_dataset_exists()
    return [path for path in DATASET_PATH if path.is_dir()]

def get_classes():
    ensure_dataset_exists()
    classes = set()
    for path in get_available_dataset_paths():
        for folder in path.iterdir():
            if folder.is_dir() and "___" in folder.name:
                classes.add(folder.name)
    return sorted(list(classes))


def get_images_by_class():
    ensure_dataset_exists()
    dataset = []

    for path in get_available_dataset_paths():
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
                sample = {
                    "image": image,
                    "label": folder_name,
                    "source": "plantvillage",
                    "group": leaf_group(folder_name, image.name),
                }
                # Máscaras de segmented só batem com as imagens originais (não com as giradas).
                if path == PRIMARY_DATASET_PATH:
                    sample["mask"] = find_mask_path(folder_name, image.name)
                dataset.append(sample)

    return dataset
