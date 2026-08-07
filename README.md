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

### Running the Project

To start the application, run:

```bash
uv run python -m leafguardai
```

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

## Dataset Exploration

The project includes a dataset exploration script to verify and analyze the downloaded dataset before training the AI model.

The script can be executed with:

```bash
uv run python scripts/explore_dataset.py
```
