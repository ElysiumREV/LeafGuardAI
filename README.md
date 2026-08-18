## Development

This project uses **uv** to manage dependencies and the virtual environment.

### Installation

After cloning the repository, run:

```bash
uv sync
```

This command will install all required dependencies and set up the project's virtual environment automatically.

### Adding Dependencies

To add a new dependency to the project, use:

```bash
uv add <package-name>
```

The `uv` tool will automatically update both the `pyproject.toml` and `uv.lock` files.

### Running the GUI

```bash
uv run python -m leafguardai
```

or:

```bash
uv run leafguardai
```

The GUI classifies a selected leaf image with the trained checkpoint in `models/`. Train the model first.

## Dataset

This project uses the PlantVillage dataset.

The dataset is not included in this repository due to its size.

You can download it from the original [repository.](https://github.com/spMohanty/PlantVillage-Dataset)
Only the image dataset is required. After downloading, place the color/ directory inside the following path:
```
LeafGuardAI/
└── data/
    └── raw/
        └── PlantVillage/
            └── color/
```
The expected structure is:
```
data/
└── raw/
    └── PlantVillage/
        └── color/
            ├── Apple___Apple_scab/
            ├── Tomato___healthy/
            └── ...
```

## Training

```bash
uv run python scripts/train.py
```

Useful flags:

```bash
uv run python scripts/train.py --epochs-stage1 10 --epochs-stage2 15 --patience 5
```

Artifacts are written to `models/`:

- `best_model.pt` — best checkpoint by validation accuracy
- `classes.json` — class index mapping
- `split.json` — stratified train/val/test paths
- `history.json` — per-epoch metrics
- `metrics.json` — final test metrics
- `classification_report.txt` — per-class precision/recall/F1
- `confusion_matrix.png`

## Inference (CLI)

```bash
uv run python scripts/predict.py caminho/da/folha.jpg --top 3
```

## Dataset Exploration

```bash
uv run python scripts/explore_dataset.py
uv run python scripts/test_dataset.py
```

## Expected Project Structure
```
LeafGuardAI/
├── src/
│   └── leafguardai/
├── scripts/
├── data/
├── models/
├── pyproject.toml
└── uv.lock
```
