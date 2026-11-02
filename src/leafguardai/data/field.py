"""Fotos de folhas tiradas em campo (vários datasets), mapeadas para as classes do PlantVillage.

Baixe com: uv run python scripts/download_field_datasets.py
Cada dataset fica em data/raw/field/<Nome>/; `FIELD_SOURCES` diz quais pastas usar e a qual
classe do PlantVillage cada uma corresponde. Pastas fora do mapeamento são ignoradas (com aviso).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIELD_ROOT = PROJECT_ROOT / "data" / "raw" / "field"
PLANTVILLAGE_PATH = PROJECT_ROOT / "data" / "raw" / "PlantVillage" / "color"
EXTERNAL_TEST_PATH = PROJECT_ROOT / "data" / "external_test"
HASH_CACHE_PATH = PROJECT_ROOT / "data" / "raw" / ".dhash_cache_v2.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Duas fotos são consideradas a mesma imagem se os DOIS dHash (horizontal e vertical) diferirem em
# até este número de bits. Um hash só gera coincidências demais ao comparar com dezenas de milhares
# de imagens (ex.: maçã "igual" a uma folha); exigir os dois elimina esses falsos positivos.
NEAR_DUPLICATE_BITS = 3

FIELD_SOURCES = {
    "PlantDoc": {
        "license": "CC BY 4.0",
        "subdirs": ["train", "test"],  # a divisão original tem duplicatas; refazemos o split
        "mapping": {
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
        },
    },
    "PlantWild": {
        "license": "CC BY-NC-ND 4.0 (uso não comercial)",
        "subdirs": [""],
        "mapping": {
            "apple black rot": "Apple___Black_rot",
            "apple leaf": "Apple___healthy",
            "apple rust": "Apple___Cedar_apple_rust",
            "apple scab": "Apple___Apple_scab",
            "bell pepper leaf": "Pepper,_bell___healthy",
            "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
            "blueberry leaf": "Blueberry___healthy",
            "cherry leaf": "Cherry_(including_sour)___healthy",
            "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
            "citrus greening disease": "Orange___Haunglongbing_(Citrus_greening)",
            "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "corn leaf": "Corn_(maize)___healthy",
            "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
            "corn rust": "Corn_(maize)___Common_rust_",
            "grape black rot": "Grape___Black_rot",
            "grape leaf": "Grape___healthy",
            "peach leaf": "Peach___healthy",
            "potato early blight": "Potato___Early_blight",
            "potato late blight": "Potato___Late_blight",
            "potato leaf": "Potato___healthy",
            "raspberry leaf": "Raspberry___healthy",
            "soybean leaf": "Soybean___healthy",
            "squash powdery mildew": "Squash___Powdery_mildew",
            "strawberry leaf": "Strawberry___healthy",
            "strawberry leaf scorch": "Strawberry___Leaf_scorch",
            "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
            "tomato early blight": "Tomato___Early_blight",
            "tomato late blight": "Tomato___Late_blight",
            "tomato leaf": "Tomato___healthy",
            "tomato leaf mold": "Tomato___Leaf_Mold",
            "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
            "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
            "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        },
    },
    "GVLiD": {
        "license": "CC BY 4.0",
        "subdirs": [""],
        "mapping": {
            "Black rot": "Grape___Black_rot",
            "esca": "Grape___Esca_(Black_Measles)",
            "healthy": "Grape___healthy",
            "leaf blight": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        },
    },
    "ESCA": {
        "license": "CC BY 4.0",
        "subdirs": [""],
        "mapping": {
            "esca": "Grape___Esca_(Black_Measles)",
            "healthy": "Grape___healthy",
        },
    },
    "PotatoEthiopia": {
        "license": "CC BY 4.0",
        "subdirs": [""],
        "mapping": {
            "Healthy": "Potato___healthy",
            "Late Blight": "Potato___Late_blight",
        },
    },
    "PlantPathology2021": {
        "license": "Apache-2.0",
        "subdirs": [""],
        "mapping": {
            # "frog eye leaf spot" é a fase da podridão-negra (Botryosphaeria obtusa) na folha.
            "frog_eye_leaf_spot": "Apple___Black_rot",
            "scab": "Apple___Apple_scab",
            "rust": "Apple___Cedar_apple_rust",
            "healthy": "Apple___healthy",
        },
    },
}


def _image_files(folder):
    return [file for file in sorted(folder.iterdir()) if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS]


def _bits_to_int(bits):
    return int("".join("1" if bit else "0" for bit in bits.flatten()), 2)


def dhash(path, size=8):
    """Par de hashes perceptuais de 64 bits (horizontal, vertical); imagens quase iguais têm
    hashes com poucos bits diferentes. Decodifica a imagem inteira, então arquivos corrompidos
    geram OSError aqui (e são descartados) em vez de derrubar o treino depois."""
    with Image.open(path) as file:
        gray = ImageOps.exif_transpose(file).convert("L").resize((size + 1, size + 1), Image.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    horizontal = pixels[:size, 1:] > pixels[:size, :-1]
    vertical = pixels[1:, :size] > pixels[:-1, :size]
    return _bits_to_int(horizontal), _bits_to_int(vertical)


def _hashes(paths):
    """Par de dHash de cada caminho (None se ilegível), com cache em data/raw/.dhash_cache.json."""
    cache = {}
    if HASH_CACHE_PATH.exists():
        cache = json.loads(HASH_CACHE_PATH.read_text(encoding="utf-8"))
    result, changed = [], False
    for path in paths:
        stat = path.stat()
        key, stamp = str(path), f"{stat.st_size}:{int(stat.st_mtime)}"
        entry = cache.get(key)
        if entry is None or entry[0] != stamp or len(entry) != 3:
            try:
                entry = [stamp, *dhash(path)]
            except OSError:
                entry = [stamp, None, None]
            cache[key] = entry
            changed = True
        result.append(None if entry[1] is None else (entry[1], entry[2]))
    if changed:
        HASH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HASH_CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return result


def _close(hashes, pair):
    """Máscara das linhas de `hashes` (array Nx2) quase iguais ao par (h, v)."""
    return ((np.bitwise_count(hashes[:, 0] ^ np.uint64(pair[0])) <= NEAR_DUPLICATE_BITS)
            & (np.bitwise_count(hashes[:, 1] ^ np.uint64(pair[1])) <= NEAR_DUPLICATE_BITS))


def _near_any(hashes, references):
    """Para cada par de hashes, True se alguma referência for quase igual."""
    if not len(references):
        return np.zeros(len(hashes), dtype=bool)
    refs = np.array(references, dtype=np.uint64)
    return np.array([_close(refs, pair).any() for pair in hashes], dtype=bool)


def get_field_samples(datasets=None, verbose=True):
    """Amostras dos datasets de campo baixados (todos, ou só os de `datasets`), com rótulo do
    PlantVillage e "group".

    - Quase-duplicatas (inclusive entre datasets diferentes) recebem o mesmo "group", para caírem
      sempre na mesma parte do split.
    - Remove fotos que também estão em data/external_test (não contaminar esse teste) e fotos que
      são cópias de imagens do PlantVillage (seriam fotos de laboratório rotuladas como de campo).
    Retorna [] se nenhum dataset de campo estiver baixado.
    """
    samples = []
    for name, source in FIELD_SOURCES.items():
        root = FIELD_ROOT / name
        if not root.is_dir() or (datasets and name not in datasets):
            continue
        count = 0
        for subdir in source["subdirs"]:
            base = root / subdir if subdir else root
            if not base.is_dir():
                continue
            for folder in sorted(base.iterdir()):
                if not folder.is_dir():
                    continue
                label = source["mapping"].get(folder.name)
                if label is None:
                    if verbose:
                        print(f"   aviso: pasta sem mapeamento ignorada: {folder}", flush=True)
                    continue
                for file in _image_files(folder):
                    samples.append({"image": file, "label": label, "source": "field", "dataset": name})
                    count += 1
        if verbose:
            print(f"   -> {name}: {count} fotos de campo", flush=True)
    if not samples:
        return []

    readable = [(s, h) for s, h in zip(samples, _hashes([s["image"] for s in samples])) if h is not None]
    if verbose and len(readable) < len(samples):
        broken = sorted({str(s["image"]) for s in samples} - {str(s["image"]) for s, _ in readable})
        print(f"   aviso: {len(broken)} imagem(ns) corrompida(s) ignorada(s), ex.: {broken[:3]}", flush=True)
    samples = [s for s, _ in readable]
    hashes = np.array([h for _, h in readable], dtype=np.uint64)

    reference_paths = []
    if EXTERNAL_TEST_PATH.is_dir():
        reference_paths += [f for d in EXTERNAL_TEST_PATH.iterdir() if d.is_dir() for f in _image_files(d)]
    external = [h for h in _hashes(reference_paths) if h is not None]
    plantvillage = []
    if PLANTVILLAGE_PATH.is_dir():
        pv_paths = [f for d in PLANTVILLAGE_PATH.iterdir() if d.is_dir() for f in _image_files(d)]
        plantvillage = [h for h in _hashes(pv_paths) if h is not None]
    drop_external = _near_any(hashes, external)
    drop_lab = _near_any(hashes, plantvillage)
    keep = ~(drop_external | drop_lab)
    if verbose:
        print(f"   -> removidas: {int(drop_external.sum())} iguais a data/external_test, "
              f"{int((drop_lab & ~drop_external).sum())} cópias do PlantVillage", flush=True)
    samples = [s for s, k in zip(samples, keep) if k]
    hashes = hashes[keep]

    # union-find para agrupar quase-duplicatas
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(samples) - 1):
        close = np.nonzero(_close(hashes[i + 1:], hashes[i]))[0]
        for j in close:
            parent[find(i)] = find(i + 1 + int(j))

    # Nome estável do grupo (o menor caminho do grupo): não muda quando outros datasets entram.
    group_names = {}
    for index, sample in enumerate(samples):
        root = find(index)
        name = str(sample["image"].relative_to(FIELD_ROOT)).replace("\\", "/")
        group_names[root] = min(group_names.get(root, name), name)
    for index, sample in enumerate(samples):
        sample["group"] = f"field:{group_names[find(index)]}"
    return samples
