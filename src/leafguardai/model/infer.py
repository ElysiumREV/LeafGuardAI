import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageOps

from leafguardai.model.classifier import create_model
from leafguardai.paths import models_dir
from leafguardai.vision.transforms import get_eval_transforms
from leafguardai.vision.processor import ImageProcessor

MODELS_DIR = models_dir()
CHECKPOINT_PATH = MODELS_DIR / "best_model.pt"
CLASSES_PATH = MODELS_DIR / "classes.json"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing.json"

# Modelo treinado publicado no GitHub Release (gerado com scripts/package_model.py).
MODEL_URL = "https://github.com/ElysiumREV/LeafGuardAI/releases/latest/download/leafguardai-model.zip"
MODEL_FILES = ("best_model.pt", "classes.json", "preprocessing.json")

# Faixas de confiança do top-1, medidas em ~4.300 fotos de campo nunca vistas no treino
# (scripts/evaluate_folder.py): acima de 70% o modelo acerta ~91%; entre 50% e 70%, ~64%;
# abaixo de 50%, ~45%. Com a planta informada, a confiança sobe sem o acerto acompanhar
# (70-90%: só ~71%; acima de 90%: ~95%), por isso o limite de "alta" é maior.
# Recalibrar a cada novo modelo.
CONFIDENCE_HIGH = 0.70
CONFIDENCE_HIGH_WITH_PLANT = 0.90
CONFIDENCE_LOW = 0.50

# Plantas que o usuário pode informar (nome exibido -> prefixo da classe). Só as que têm duas
# ou mais classes no modelo: nas outras (ex.: Soybean, só "healthy"), restringir a resposta
# daria sempre 100% na única classe conhecida, mesmo em uma folha doente.
PLANT_CHOICES = {
    "Batata": "Potato",
    "Cereja": "Cherry_(including_sour)",
    "Maçã": "Apple",
    "Milho": "Corn_(maize)",
    "Morango": "Strawberry",
    "Pêssego": "Peach",
    "Pimentão": "Pepper,_bell",
    "Tomate": "Tomato",
    "Uva": "Grape",
}


def confidence_level(probability, plant_known=False):
    """'alta', 'media' ou 'baixa', conforme as faixas acima."""
    if probability >= (CONFIDENCE_HIGH_WITH_PLANT if plant_known else CONFIDENCE_HIGH):
        return "alta"
    if probability >= CONFIDENCE_LOW:
        return "media"
    return "baixa"


def model_available():
    return CHECKPOINT_PATH.exists() and CLASSES_PATH.exists()


MAX_PACKAGE_BYTES = 500 * 1024 * 1024  # limite de segurança para o zip e o conteúdo extraído


class ModelDownloadError(FileNotFoundError):
    """Falha ao baixar ou validar o modelo (herda de FileNotFoundError por compatibilidade)."""


def _download_file(url, destination, progress=None):
    with urllib.request.urlopen(url, timeout=60) as response, open(destination, "wb") as file:
        total = int(response.headers.get("Content-Length") or 0)
        if total > MAX_PACKAGE_BYTES:
            raise ModelDownloadError(f"O pacote em {url} é grande demais ({total / 1e6:.0f} MB).")
        done = 0
        while chunk := response.read(256 * 1024):
            done += len(chunk)
            if done > MAX_PACKAGE_BYTES:
                raise ModelDownloadError(f"O pacote em {url} é grande demais.")
            file.write(chunk)
            if progress:
                progress(done, total)


def _expected_sha256(url):
    """Lê o hash esperado de `<url>.sha256` (formato do sha256sum: '<hash>  <arquivo>')."""
    with urllib.request.urlopen(url + ".sha256", timeout=60) as response:
        text = response.read(4096).decode("utf-8", errors="replace").strip()
    token = text.split()[0].lower() if text else ""
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ModelDownloadError(f"O arquivo {url}.sha256 não contém um SHA-256 válido.")
    return token


def _sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(url=None, progress=None):
    """Baixa o modelo do GitHub Release, confere o SHA-256 e extrai os arquivos em models/.

    `progress(baixado, total)` é chamado durante o download (total é 0 se o servidor não informar).
    O hash vem de `<url>.sha256`, publicado junto do zip por scripts/package_model.py: ele
    detecta download corrompido ou incompleto; não protege contra um release adulterado,
    já que o hash é publicado no mesmo lugar do zip.
    """
    url = url or MODEL_URL
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "model.zip"
        try:
            expected = _expected_sha256(url)
            _download_file(url, archive, progress)
        except ModelDownloadError:
            raise
        except OSError as error:
            raise ModelDownloadError(
                f"Não encontrei um modelo em {MODELS_DIR} e não consegui baixá-lo de {url} ({error}). "
                "Verifique a conexão ou rode scripts/train.py para treinar um modelo."
            ) from error

        actual = _sha256_of(archive)
        if actual != expected:
            raise ModelDownloadError(
                "O modelo baixado não confere com o SHA-256 publicado (download corrompido ou "
                "release alterado). Nada foi instalado; tente novamente."
            )

        staging = Path(temp) / "staging"
        staging.mkdir()
        with zipfile.ZipFile(archive) as package:
            sizes = {info.filename: info.file_size for info in package.infolist()}
            missing = [name for name in MODEL_FILES if name not in sizes]
            if missing:
                raise ModelDownloadError(f"O pacote baixado não contém: {', '.join(missing)}")
            if sum(sizes[name] for name in MODEL_FILES) > MAX_PACKAGE_BYTES:
                raise ModelDownloadError("O conteúdo do pacote é grande demais.")
            # Só os nomes esperados são extraídos, direto na pasta temporária (sem caminhos do zip).
            for name in MODEL_FILES:
                with package.open(name) as source, open(staging / name, "wb") as target:
                    shutil.copyfileobj(source, target)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES:
            shutil.move(str(staging / name), MODELS_DIR / name)


def ensure_model():
    """Garante que existe um modelo em models/, baixando o do GitHub Release se preciso."""
    if not model_available():
        download_model()


def format_class_name(name: str) -> str:
    """
    Formats class names from 'Planta___Doenca' to a human-readable format.
    Example: 'Apple___Apple_scab' -> 'Apple Scab'
    """
    if "___" not in name:
        return name.replace("_", " ").capitalize()

    # Split into plant and disease
    plant, disease = name.split("___", 1)
    plant = plant.replace("_", " ")  # ex.: 'Pepper,_bell' -> 'Pepper, bell'

    # Clean disease name: replace underscores with spaces
    disease_clean = disease.replace("_", " ")

    # Avoid redundancy: If disease name starts with plant name, remove it
    # Example: 'Apple_scab' -> 'Scab' when plant is 'Apple'
    if disease_clean.lower().startswith(plant.lower()):
        # Remove plant name and any trailing underscores/spaces
        disease_clean = disease_clean[len(plant):].strip(" _-")

    # Capitalize words
    final_name = f"{plant} {disease_clean}".strip()
    return " ".join(word.capitalize() for word in final_name.split())


def load_classes():
    ensure_model()
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CLASSES_PATH}. Rode scripts/train.py primeiro "
            "(o treino salva classes.json junto com o checkpoint)."
        )

    with open(CLASSES_PATH, encoding="utf-8") as file:
        return json.load(file)


def load_model(num_classes, device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrei {CHECKPOINT_PATH}. Rode scripts/train.py primeiro."
        )

    model = create_model(num_classes=num_classes)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def uses_processor():
    """Se a inferência deve aplicar o ImageProcessor.

    scripts/train.py grava models/preprocessing.json com o pré-processamento usado no
    treino, e a inferência repete exatamente o mesmo (usar um diferente derruba a acurácia:
    99% -> 64% em folhas do PlantVillage). Checkpoints antigos, sem esse arquivo, mantêm o
    comportamento original do app (com processor).
    """
    if not PREPROCESSING_PATH.exists():
        return True
    with open(PREPROCESSING_PATH, encoding="utf-8") as file:
        return bool(json.load(file).get("use_processor", False))


def preprocess_image(image_path, use_processor=None):
    if use_processor is None:
        use_processor = uses_processor()
    if use_processor:
        image = ImageProcessor().process(image_path)
    else:
        with Image.open(image_path) as file:
            image = ImageOps.exif_transpose(file).convert("RGB")

    transform = get_eval_transforms()
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict(image_path, top_k=3, model=None, classes=None, device=None, plant=None):
    """Top-k de (classe, probabilidade). Com `plant` (prefixo da classe, ex.: 'Tomato'), considera
    só as classes dessa planta e redistribui as probabilidades entre elas."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if classes is None:
        classes = load_classes()

    if model is None:
        model = load_model(num_classes=len(classes), device=device)

    input_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    available = len(classes)
    if plant is not None:
        mask = torch.tensor([name.split("___", 1)[0] == plant for name in classes], device=device)
        available = int(mask.sum())
        if available == 0:
            raise ValueError(f"O modelo não tem classes da planta '{plant}'.")
        probabilities = probabilities * mask
        probabilities = probabilities / probabilities.sum()

    top_probs, top_indices = probabilities.topk(min(top_k, available))

    return [
        (classes[index.item()], prob.item())
        for prob, index in zip(top_probs, top_indices)
    ]
