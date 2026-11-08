"""Baixa os datasets de fotos de campo para data/raw/field/.

Uso:
    uv run python scripts/download_field_datasets.py            # todos
    uv run python scripts/download_field_datasets.py --only plantdoc gvlid

Cada dataset vai para data/raw/field/<Nome>/<pasta da classe>/; o mapeamento das pastas
para as classes do PlantVillage fica em src/leafguardai/data/field.py. Fontes e licenças
estão no README.
"""
import argparse
import io
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from leafguardai.data.field import FIELD_ROOT, FIELD_SOURCES

INVALID_CHARS = re.compile(r'[<>:"|?*]')
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/{}"
USER_AGENT = {"User-Agent": "LeafGuardAI dataset downloader"}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120) as response:
        return response.read()


def _download(url, destination):
    request = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as file:
        done, step = 0, 100 * 1024**2
        while chunk := response.read(1024 * 1024):
            file.write(chunk)
            done += len(chunk)
            if done // step != (done - len(chunk)) // step:
                print(f"   baixado: {done / 1024**2:,.0f} MB", flush=True)


def _mendeley_files(dataset_id):
    return json.loads(_get(MENDELEY_API.format(dataset_id)))["files"]


def _extract_rar(archive, destination):
    """bsdtar (padrão no Windows 10+ e no macOS) lê RAR; senão tenta unrar ou 7z."""
    candidates = [
        ["C:/Windows/System32/tar.exe", "-xf", str(archive), "-C", str(destination)],
        ["bsdtar", "-xf", str(archive), "-C", str(destination)],
        ["unrar", "x", "-o+", str(archive), str(destination) + "/"],
        ["7z", "x", "-y", f"-o{destination}", str(archive)],
    ]
    for command in candidates:
        if shutil.which(command[0]) or Path(command[0]).exists():
            if subprocess.run(command, capture_output=True).returncode == 0:
                return
    raise RuntimeError(f"Não consegui extrair {archive}: instale bsdtar, unrar ou 7-Zip.")


def _move_class_dirs(source_root, target, folders):
    target.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        source = source_root / folder
        if source.is_dir():
            shutil.move(str(source), str(target / folder))


def download_plantdoc(target):
    url = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "plantdoc.zip"
        _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                parts = member.filename.split("/")[1:]  # remove "PlantDoc-Dataset-master/"
                if member.is_dir() or len(parts) < 3 or parts[0] not in {"train", "test"}:
                    continue
                path = target.joinpath(*(INVALID_CHARS.sub("_", part) for part in parts))
                path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(path, "wb") as destination:
                    shutil.copyfileobj(source, destination)


def download_plantwild(target):
    url = "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip"
    wanted = set(FIELD_SOURCES["PlantWild"]["mapping"])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "plantwild.zip"
        _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                parts = member.filename.split("/")
                # plantwild/images/<classe>/<arquivo>
                if member.is_dir() or len(parts) != 4 or parts[2] not in wanted:
                    continue
                path = target / parts[2] / INVALID_CHARS.sub("_", parts[3])
                path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(path, "wb") as destination:
                    shutil.copyfileobj(source, destination)


def _download_mendeley_rar(dataset_id, target, inner_root, folders):
    file = _mendeley_files(dataset_id)[0]
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / file["filename"]
        _download(file["content_details"]["download_url"], archive)
        _extract_rar(archive, Path(tmp))
        _move_class_dirs(Path(tmp) / inner_root, target, folders)


def download_gvlid(target):
    _download_mendeley_rar("wkymf8bhcg", target, "DATASET", FIELD_SOURCES["GVLiD"]["mapping"])


def download_potato_ethiopia(target):
    _download_mendeley_rar("v4w72bsts5", target, "Local", FIELD_SOURCES["PotatoEthiopia"]["mapping"])


def download_esca(target):
    # Rótulo no nome do arquivo: esca_000_cam1.jpg / healthy_000_cam1.jpg
    files = [f for f in _mendeley_files("89cnxc58kj") if f["filename"].lower().endswith(".jpg")]

    def fetch(file):
        label = file["filename"].split("_", 1)[0]
        path = target / label / file["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(_get(file["content_details"]["download_url"]))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, files))


def download_plant_pathology_2021(target, per_class=500, max_side=1024):
    # Fotos de pomar (Cornell AgriTech). Só imagens com um único rótulo, até `per_class` por classe,
    # reduzidas para no máximo `max_side` pixels (as originais têm 4000x2672).
    wanted = set(FIELD_SOURCES["PlantPathology2021"]["mapping"])
    counts = {label: 0 for label in wanted}
    api = ("https://datasets-server.huggingface.co/rows?dataset=timm%2Fplant-pathology-2021"
           "&config=default&split=train&offset={}&length=100")

    def fetch(job):
        url, path = job
        image = Image.open(io.BytesIO(_get(url))).convert("RGB")
        image.thumbnail((max_side, max_side))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, quality=92)

    offset = 0
    while any(count < per_class for count in counts.values()):
        page = json.loads(_get(api.format(offset)))
        rows = page.get("rows", [])
        if not rows:
            break
        jobs = []
        for item in rows:
            names = item["row"]["label_names"]
            if len(names) != 1 or names[0] not in wanted or counts[names[0]] >= per_class:
                continue
            path = target / names[0] / f"{item['row']['image_id']}.jpg"
            if not path.exists():
                jobs.append((item["row"]["image"]["src"], path))
            counts[names[0]] += 1
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(fetch, jobs))
        offset += len(rows)
        print(f"   {offset} linhas lidas | {counts}", flush=True)


def download_tomato_pakistan(target):
    # O zip tem 830 fotos originais em "Dataset (raw)" e ~7.200 cópias aumentadas (ignoradas:
    # o treino já faz aumento de dados, e cópias da mesma foto vazariam entre treino e teste).
    file = _mendeley_files("3mbnb82mxd")[0]
    wanted = set(FIELD_SOURCES["TomatoPakistan"]["mapping"])
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "tomato_pakistan.zip"
        _download(file["content_details"]["download_url"], archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                parts = member.filename.split("/")
                # Tomato Dataset/Dataset (raw)/<classe>/<arquivo>
                if member.is_dir() or len(parts) != 4 or parts[1] != "Dataset (raw)" or parts[2] not in wanted:
                    continue
                path = target / parts[2] / INVALID_CHARS.sub("_", parts[3])
                path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as source, open(path, "wb") as destination:
                    shutil.copyfileobj(source, destination)


def download_tomato_diu(target):
    # Arquivos soltos no Mendeley; a classe é o nome da pasta de cada arquivo.
    dataset_id = "93h9p62kg4"
    version = json.loads(_get(MENDELEY_API.format(dataset_id)))["version"]
    folders = {f["id"]: f["name"] for f in json.loads(_get(f"{MENDELEY_API.format(dataset_id)}/folders/{version}"))}
    wanted = set(FIELD_SOURCES["TomatoDIU"]["mapping"])
    files = [f for f in _mendeley_files(dataset_id)
             if folders.get(f.get("folder_id")) in wanted and f["filename"].lower().endswith(".jpg")]

    def fetch(file):
        path = target / folders[file["folder_id"]] / INVALID_CHARS.sub("_", file["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(_get(file["content_details"]["download_url"]))

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, files))


DOWNLOADERS = {
    "PlantDoc": download_plantdoc,
    "PlantWild": download_plantwild,
    "GVLiD": download_gvlid,
    "ESCA": download_esca,
    "PotatoEthiopia": download_potato_ethiopia,
    "PlantPathology2021": download_plant_pathology_2021,
    "TomatoPakistan": download_tomato_pakistan,
    "TomatoDIU": download_tomato_diu,
}


def main():
    parser = argparse.ArgumentParser(description="Baixa os datasets de fotos de campo.")
    parser.add_argument("--only", nargs="+", choices=[n.lower() for n in DOWNLOADERS],
                        help="Baixa só os datasets indicados.")
    args = parser.parse_args()

    for name, downloader in DOWNLOADERS.items():
        if args.only and name.lower() not in args.only:
            continue
        target = FIELD_ROOT / name
        if target.is_dir() and any(target.iterdir()):
            print(f"{name}: já existe em {target} (apague a pasta para baixar de novo).")
            continue
        print(f"{name}: baixando ({FIELD_SOURCES[name]['license']})...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        try:
            downloader(target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        total = sum(1 for _ in target.rglob("*") if _.is_file())
        print(f"{name}: {total} imagens em {target}", flush=True)


if __name__ == "__main__":
    main()
