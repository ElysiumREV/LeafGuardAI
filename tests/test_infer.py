import json

import pytest
import torch

from leafguardai.model import infer
from leafguardai.model.infer import (
    CONFIDENCE_HIGH,
    CONFIDENCE_HIGH_WITH_PLANT,
    CONFIDENCE_LOW,
    PLANT_CHOICES,
    confidence_level,
    format_class_name,
    predict,
)


@pytest.mark.parametrize(
    "probability, plant_known, expected",
    [
        (0.99, False, "alta"),
        (CONFIDENCE_HIGH, False, "alta"),
        (CONFIDENCE_HIGH - 0.01, False, "media"),
        (CONFIDENCE_LOW, False, "media"),
        (CONFIDENCE_LOW - 0.01, False, "baixa"),
        # com a planta informada o limite de "alta" sobe
        (CONFIDENCE_HIGH, True, "media"),
        (CONFIDENCE_HIGH_WITH_PLANT, True, "alta"),
        (0.2, True, "baixa"),
    ],
)
def test_confidence_level(probability, plant_known, expected):
    assert confidence_level(probability, plant_known=plant_known) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Apple___Apple_scab", "Apple Scab"),
        ("Tomato___Late_blight", "Tomato Late Blight"),
        ("Pepper,_bell___healthy", "Pepper, Bell Healthy"),
        ("Corn_(maize)___Common_rust_", "Corn (maize) Common Rust"),
        ("sem_separador", "Sem separador"),
    ],
)
def test_format_class_name(raw, expected):
    assert format_class_name(raw) == expected


CLASSES = ["Apple___scab", "Apple___healthy", "Tomato___blight", "Tomato___healthy"]


@pytest.fixture
def fake_inference(monkeypatch):
    """Modelo falso que sempre favorece 'Apple___scab', sem pré-processamento real."""
    monkeypatch.setattr(infer, "preprocess_image", lambda path: torch.zeros(1, 3, 2, 2))
    logits = torch.tensor([[5.0, 1.0, 4.0, 0.0]])
    return lambda tensor: logits


def run(fake_model, **kwargs):
    return predict("x.jpg", model=fake_model, classes=CLASSES, device=torch.device("cpu"), **kwargs)


def test_predict_returns_sorted_top_k(fake_inference):
    results = run(fake_inference, top_k=3)
    assert [name for name, _ in results] == ["Apple___scab", "Tomato___blight", "Apple___healthy"]
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)


def test_predict_plant_filter_restricts_and_renormalizes(fake_inference):
    results = run(fake_inference, top_k=5, plant="Tomato")
    assert {name for name, _ in results} == {"Tomato___blight", "Tomato___healthy"}
    assert sum(p for _, p in results) == pytest.approx(1.0)
    assert results[0][0] == "Tomato___blight"


def test_predict_unknown_plant_raises(fake_inference):
    with pytest.raises(ValueError):
        run(fake_inference, plant="Banana")


def test_plant_choices_match_trained_classes():
    classes_path = infer.CLASSES_PATH
    if not classes_path.exists():
        pytest.skip("modelo não está instalado")
    prefixes = {name.split("___", 1)[0] for name in json.loads(classes_path.read_text(encoding="utf-8"))}
    assert set(PLANT_CHOICES.values()) <= prefixes
