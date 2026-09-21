# LeafGuardAI 🌿

LeafGuardAI is a deep learning project focused on identifying plant diseases through leaf image analysis, combining advanced computer vision preprocessing with a fine-tuned EfficientNet-B0 architecture.

## 🛠️ Image Processing Pipeline

The image processing pipeline is designed to maximize the extraction of relevant leaf features while minimizing interference from noise and lighting variations. The flow consists of the following sequential stages:

1.  **Leaf Segmentation (ROI Extraction):** 
    *   Conversion from BGR to **HSV** color space.
    *   Application of a color mask to isolate green tones.
    *   Morphological operations (**Closing** and **Opening**) with an elliptical kernel to refine the mask.
    *   Identification of the largest contour and cropping to focus exclusively on the leaf, removing irrelevant backgrounds.

2.  **Chromatic Correction and Balancing:** 
    *   Estimation of lighting bias by calculating the average color of background pixels (outside the leaf mask).
    *   Application of a correction factor to normalize image colors, reducing dependency on specific lighting conditions.

3.  **Denoising:** 
    *   Utilization of the `fastNlMeansDenoisingColored` algorithm to remove granular noise while preserving edges and texture details.

4.  **Contrast Optimization (CLAHE):** 
    *   Conversion to **Lab** color space.
    *   **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) applied to the luminosity channel ($L$) to enhance local contrast and highlight subtle disease patterns.

5.  **Sharpness Enhancement (Unsharp Masking):** 
    *   Application of a Gaussian blur combined with the original image to create a sharpening effect, emphasizing edges and leaf vein structures.

6.  **Dimensional Normalization:** 
    *   Final resize to **256x256 pixels** using `INTER_AREA` interpolation for compatibility with the neural network input.

## 🤖 AI Implementation & Integration

The project integrates advanced AI concepts, focusing on **Deep Learning** and **Computer Vision** for botanical pathology classification.

### Full Project Pipeline:
**A. Data Management:**
*   **Stratified Split:** Data is divided into **Train, Validation, and Test** sets while maintaining class distribution.
*   **Class Balancing:** Use of **class weights** in the loss function to handle imbalanced datasets, penalizing errors in minority classes more heavily.

**B. Model Architecture:**
*   **Base:** **EfficientNet-B0**, chosen for its optimal balance between depth, width, and resolution.
*   **Transfer Learning:** The model uses weights pre-trained on ImageNet to leverage general feature recognition (shapes and textures).

**C. Training Strategy (Multi-stage):**
1.  **Stage 1 (Warm-up):** Backbone frozen; only the classifier head is trained to avoid destroying pre-trained weights with random gradients.
2.  **Stage 2 (Fine-Tuning):** Last blocks of the backbone are unfrozen and trained with a significantly lower learning rate ($\text{lr} \approx 10^{-5}$) for specialization in leaf patterns.
*   **Optimization:** **Adam** optimizer with **ReduceLROnPlateau** scheduler.
*   **Regularization:** **Label Smoothing** ($0.1$) to prevent overfitting and improve generalization.

**D. Acceleration and Stability:**
*   **AMP (Automatic Mixed Precision):** Used to reduce VRAM consumption and accelerate GPU processing.
*   **Early Stopping:** Monitors validation accuracy to halt training when improvement ceases, preventing overtraining.

**E. Evaluation Metrics:**
*   **Confusion Matrix:** To analyze inter-class confusion.
*   **Classification Report:** Calculation of **Precision**, **Recall**, and **Macro-F1 Score**.
*   **Top-K Accuracy:** Evaluates if the correct class is among the top $K$ predictions.

---

## 🚀 Development

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

## 📁 Dataset

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

## 🏋️ Training

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

### Training with AMD GPU (ROCm + Docker)

On Linux with an AMD GPU supported by ROCm, the project can be trained in the
official ROCm/PyTorch container. The host must expose `/dev/kfd` and `/dev/dri`
and use the native Docker Engine (not Docker Desktop's `desktop-linux` context).

Build the image once:

```bash
docker --context default compose build
```

Train using the GPU:

```bash
docker --context default compose run --rm train
```

The current directory is mounted at `/workspace`; therefore the dataset under
`data/` and artifacts generated in `models/` remain on the host. Extra training
arguments can be supplied after the service name:

```bash
docker --context default compose run --rm train --batch-size 64 --epochs-stage1 10 --epochs-stage2 15
```

## 🔍 Inference (CLI)

```bash
uv run python scripts/predict.py caminho/da/folha.jpg --top 3
```

## 📊 Dataset Exploration

```bash
uv run python scripts/explore_dataset.py
uv run python scripts/test_dataset.py
```

## 🏗️ Expected Project Structure
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
