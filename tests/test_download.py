import hashlib
import zipfile

import pytest

from leafguardai.model import infer
from leafguardai.model.infer import ModelDownloadError, download_model


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Um 'release' local (file://) com o zip do modelo e o .sha256, e MODELS_DIR em tmp."""
    server = tmp_path / "server"
    server.mkdir()
    archive = server / "model.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for name in infer.MODEL_FILES:
            package.writestr(name, f"conteudo de {name}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (server / "model.zip.sha256").write_text(f"{digest}  model.zip\n", encoding="utf-8")

    models = tmp_path / "models"
    monkeypatch.setattr(infer, "MODELS_DIR", models)
    return archive.as_uri(), server, models


def test_download_installs_files_and_reports_progress(release):
    url, _, models = release
    calls = []
    download_model(url, progress=lambda done, total: calls.append(done))
    for name in infer.MODEL_FILES:
        assert (models / name).read_text(encoding="utf-8") == f"conteudo de {name}"
    assert calls and calls[-1] > 0


def test_download_rejects_hash_mismatch_and_installs_nothing(release):
    url, server, models = release
    (server / "model.zip.sha256").write_text("0" * 64 + "  model.zip\n", encoding="utf-8")
    with pytest.raises(ModelDownloadError, match="SHA-256"):
        download_model(url)
    assert not models.exists()


def test_download_rejects_malformed_hash_file(release):
    url, server, _ = release
    (server / "model.zip.sha256").write_text("nao-e-um-hash\n", encoding="utf-8")
    with pytest.raises(ModelDownloadError, match="SHA-256 válido"):
        download_model(url)


def test_download_without_hash_file_fails(release):
    url, server, models = release
    (server / "model.zip.sha256").unlink()
    with pytest.raises(FileNotFoundError):
        download_model(url)
    assert not models.exists()


def test_download_rejects_package_missing_files(release):
    url, server, models = release
    archive = server / "model.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("best_model.pt", "x")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (server / "model.zip.sha256").write_text(digest, encoding="utf-8")
    with pytest.raises(ModelDownloadError, match="não contém"):
        download_model(url)
    assert not models.exists()
