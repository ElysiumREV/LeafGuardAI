from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = Path(
  f"{PROJECT_ROOT}/data/raw/PlantVillage/color/"
)

def get_classes():
    return [
        folder.name
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
    ]

def get_images_by_class():
    dataset = []

    for folder in DATASET_PATH.iterdir():

        if folder.is_dir():
            images = [
                file
                for file in folder.iterdir()
                if file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]
            ]

            for image in images:
                dataset.append(
                    {
                        "image": image,
                        "label": folder.name
                    }
                )

    return dataset
