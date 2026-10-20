"""PlantDoc: fotos de folhas tiradas em campo, mapeadas para as classes do PlantVillage.

Fonte: Singh et al., "PlantDoc: A Dataset for Visual Plant Disease Detection" (2020),
https://github.com/pratikkayal/PlantDoc-Dataset — licença CC BY 4.0.
Baixe com: uv run python scripts/download_plantdoc.py
"""
from pathlib import Path

from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLANTDOC_PATH = PROJECT_ROOT / "data" / "raw" / "PlantDoc"
EXTERNAL_TEST_PATH = PROJECT_ROOT / "data" / "external_test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Duas fotos são consideradas a mesma imagem se o dHash diferir em até este número de bits.
NEAR_DUPLICATE_BITS = 6

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


def dhash(path, size=8):
    """Hash perceptual de 64 bits; imagens quase iguais têm hashes com poucos bits diferentes."""
    with Image.open(path) as file:
        file.draft("L", (size * 8, size * 8))
        gray = ImageOps.exif_transpose(file).convert("L").resize((size + 1, size), Image.BILINEAR)
    pixels = list(gray.getdata())
    value = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            value = (value << 1) | (right > left)
    return value


def _is_near(a, b):
    return bin(a ^ b).count("1") <= NEAR_DUPLICATE_BITS


def _image_files(folder):
    return [file for file in sorted(folder.iterdir()) if file.suffix.lower() in IMAGE_EXTENSIONS]


def get_plantdoc_samples():
    """Amostras do PlantDoc com rótulo do PlantVillage e um "group" por foto.

    - Ignora a divisão train/test original do PlantDoc (ela tem duplicatas entre as partes);
      quase-duplicatas recebem o mesmo "group" para caírem sempre na mesma parte do split.
    - Remove fotos que também estão em data/external_test, para não contaminar esse teste.
    Retorna [] se o PlantDoc não estiver baixado.
    """
    if not PLANTDOC_PATH.is_dir():
        return []

    samples = []
    for part in ("train", "test"):
        part_dir = PLANTDOC_PATH / part
        if not part_dir.is_dir():
            continue
        for folder in sorted(part_dir.iterdir()):
            label = PLANTDOC_TO_PLANTVILLAGE.get(folder.name)
            if not folder.is_dir() or label is None:
                continue
            for file in _image_files(folder):
                try:
                    samples.append({"image": file, "label": label, "source": "plantdoc", "hash": dhash(file)})
                except OSError:
                    continue

    external_hashes = []
    if EXTERNAL_TEST_PATH.is_dir():
        for folder in EXTERNAL_TEST_PATH.iterdir():
            if folder.is_dir():
                for file in _image_files(folder):
                    try:
                        external_hashes.append(dhash(file))
                    except OSError:
                        continue
    samples = [s for s in samples if not any(_is_near(s["hash"], h) for h in external_hashes)]

    # union-find para agrupar quase-duplicatas
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if _is_near(samples[i]["hash"], samples[j]["hash"]):
                parent[find(i)] = find(j)

    for index, sample in enumerate(samples):
        sample["group"] = f"plantdoc:{find(index)}"
        del sample["hash"]
    return samples
