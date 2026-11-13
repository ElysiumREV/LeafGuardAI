from pathlib import Path

from leafguardai import paths


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.MODELS_DIR_ENV, str(tmp_path / "custom"))
    assert paths.models_dir() == tmp_path / "custom"


def test_source_checkout_uses_repo_models(monkeypatch):
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "_is_source_checkout", lambda: True)
    assert paths.models_dir() == paths._REPO_ROOT / "models"


def test_installed_package_uses_user_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.MODELS_DIR_ENV, raising=False)
    monkeypatch.setattr(paths, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.models_dir() == Path(tmp_path) / "leafguardai" / "models"
