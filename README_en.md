# LeafGuardAI 🌿

LeafGuardAI is a deep learning project focused on identifying plant diseases through leaf image analysis, using a fine-tuned EfficientNet-B0 architecture trained on lab photos (PlantVillage) and real-world field photos. The project also includes an optional computer vision preprocessing pipeline (`ImageProcessor`).

## 🛠️ Image Processing Pipeline

### Default pipeline (training and inference)

By default, every image goes through the same steps in training and in the app (`src/leafguardai/vision/transforms.py` and `src/leafguardai/model/infer.py`):

1.  **Loading:** the image is opened as RGB, fixing the rotation of phone photos (EXIF orientation).
2.  **Resizing:** the shorter side is resized to **256 pixels**, keeping the aspect ratio.
3.  **Center crop:** a **224x224 pixel** crop, the EfficientNet-B0 input size.
4.  **Normalization:** RGB channels are normalized with the ImageNet mean and standard deviation.

During training, steps 2 and 3 are replaced by random variations (data augmentation): crops of varying size and position, horizontal flips, rotations of up to 15° and brightness, contrast and saturation changes.

### `ImageProcessor` (optional)

The `ImageProcessor` (`src/leafguardai/vision/processor.py`) is an enhancement pipeline designed to highlight leaf features and reduce noise and lighting variations. When enabled with `--use-processor` in training, it is applied before the steps above, both in training and in the app. The flow consists of the following sequential stages:

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
    *   Final resize to **256x256 pixels** using `INTER_AREA` interpolation; the evaluation transforms then center-crop to **224x224** and normalize the colors for the neural network input.

> **Important:** the `ImageProcessor` must be used the same way in training and inference. The model learns from images in the form it saw them; applying the processor only at inference dropped accuracy from 99% to 64% on PlantVillage leaves. Training therefore records in `preprocessing.json` whether the processor was used (`--use-processor`), and the app automatically repeats the same choice. In our tests with field photos, training without the processor gave higher accuracy (about 60% versus 49%) and makes analysis faster, so it is off by default. Older checkpoints, without `preprocessing.json`, are still used with the processor.

## 🤖 AI Implementation & Integration

The project integrates advanced AI concepts, focusing on **Deep Learning** and **Computer Vision** for botanical pathology classification.

### Full Project Pipeline:
**A. Data Management:**
*   **Stratified Split by Leaf:** PlantVillage photos are divided into **Train, Validation, and Test** sets (70% / 15% / 15%) while maintaining class distribution; copies and rotated versions of the same photo always stay in the same set.
*   **Field Photos:** Besides PlantVillage (lab photos), training uses real-world photos from several field datasets, split separately (60% / 15% / 25%). The split is stable: each photo always lands in the same set, even when other datasets are added. Photos repeated across datasets are grouped, and copies of PlantVillage images, photos identical to those in `data/external_test` and corrupted files are discarded.
*   **Per-epoch Mix:** Since there are far fewer field photos, they are resampled to make up about **40%** of each epoch (`--field-share`).
*   **Class Balancing:** Use of **class weights** in the loss function to handle imbalanced datasets, penalizing errors in minority classes more heavily.

**B. Model Architecture:**
*   **Base:** **EfficientNet-B0**, chosen for its optimal balance between depth, width, and resolution.
*   **Transfer Learning:** The model uses weights pre-trained on ImageNet to leverage general feature recognition (shapes and textures).

**C. Training Strategy (Multi-stage):**
1.  **Stage 1 (Warm-up):** Backbone frozen; only the classifier head is trained ($\text{lr} = 10^{-3}$, up to 5 epochs) to avoid destroying pre-trained weights with random gradients.
2.  **Stage 2 (Fine-Tuning):** The whole network (all 9 backbone blocks) is unfrozen and trained with a learning rate of $\text{lr} = 10^{-4}$ for up to 25 epochs, adapting every layer to leaf photos, including field photos. Use `--unfreeze-blocks` to unfreeze fewer blocks.
*   **Optimization:** **Adam** optimizer with **ReduceLROnPlateau** scheduler.
*   **Regularization:** **Label Smoothing** ($0.1$) and data augmentation (crops, flips, rotations and color changes) to prevent overfitting and improve generalization.

**D. Acceleration and Stability:**
*   **AMP (Automatic Mixed Precision):** Used to reduce VRAM consumption and accelerate GPU processing.
*   **Early Stopping:** Monitors validation accuracy (on field photos, when they are present) and stops each stage after 5 epochs without improvement, preventing overtraining. The best checkpoint is the one with the highest accuracy on that validation set.

**E. Evaluation Metrics:**
*   **Confusion Matrix:** To analyze inter-class confusion.
*   **Classification Report:** Calculation of **Precision**, **Recall**, and **Macro-F1 Score**.
*   **Top-K Accuracy:** Evaluates if the correct class is among the top $K$ predictions.
*   **Field Photo Test:** Besides the PlantVillage test, the model is evaluated separately on the field test photos, which better represent real use of the app.

---

## 🚀 Development

This project uses **uv** to manage dependencies and the virtual environment.

### Installation

After cloning the repository, run:

```bash
uv sync
```

This command will install all required dependencies and set up the project's virtual environment automatically. On Windows and Linux, the installed PyTorch already supports NVIDIA GPUs (CUDA); without a GPU, it runs normally on the CPU.

### Adding Dependencies

To add a new dependency to the project, use:

```bash
uv add <package-name>
```

The `uv` tool will automatically update both the `pyproject.toml` and `uv.lock` files.

### Setting up the Gemini API

Create a key in [Google AI Studio](https://aistudio.google.com/api-keys) and paste it into the "Insira a API Key" field at the bottom of the interface.

### Running the GUI

```bash
uv run python -m leafguardai
```

or:

```bash
uv run leafguardai
```

The GUI classifies a selected leaf image with the model in `models/`. No training is needed: if `models/` has no model, the app downloads the trained model from the [latest release](https://github.com/ElysiumREV/LeafGuardAI/releases/latest) on the first analysis (about 16 MB). The command line (`scripts/predict.py`) does the same.

The app shows the 3 most likely classes and warns when confidence is low. Below 70%, the result is shown as "moderate confidence"; below 50%, as "uncertain result", and in that case the app does not ask Gemini for treatment guidance and suggests taking another photo. The thresholds are `CONFIDENCE_HIGH` and `CONFIDENCE_LOW` (`src/leafguardai/model/infer.py`), measured on field photos: above 70%, the model is right about 90% of the time.

## 📁 Dataset

This project uses **PlantVillage** (leaf photos taken in a lab) and a set of **field leaf photo datasets**.

The datasets are not included in this repository due to their size.

You can download PlantVillage from the original [repository.](https://github.com/spMohanty/PlantVillage-Dataset)
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

Optionally, NewPlantDataset (a version of PlantVillage with rotated and flipped copies) is also used if it is at `data/raw/NewPlantDataset/train/`.

The field photo datasets can be downloaded from their original repositories with the script below, which already places the images in the correct path (use `--only` to download just some of them, e.g. `--only plantdoc gvlid`):
```bash
uv run python scripts/download_field_datasets.py
```
```
LeafGuardAI/
└── data/
    └── raw/
        └── field/
```
The expected structure is:
```
data/
└── raw/
    └── field/
        ├── PlantDoc/
        │   ├── train/
        │   │   ├── Apple Scab Leaf/
        │   │   └── ...
        │   └── test/
        ├── PlantWild/
        │   ├── apple black rot/
        │   └── ...
        ├── GVLiD/
        ├── ESCA/
        ├── PotatoEthiopia/
        ├── PlantPathology2021/
        ├── TomatoPakistan/
        └── TomatoDIU/
```

Each dataset's folders are mapped to the PlantVillage classes in `src/leafguardai/data/field.py`. Sources:

- **PlantDoc** — Singh, D. et al. *PlantDoc: A Dataset for Visual Plant Disease Detection*. CODS-COMAD 2020. [Repository](https://github.com/pratikkayal/PlantDoc-Dataset) (CC BY 4.0).
- **PlantWild** — Wei, T. et al. *Benchmarking In-the-wild Multimodal Plant Disease Recognition and A Versatile Baseline*. ACM MM 2024. [Hugging Face](https://huggingface.co/datasets/uqtwei2/PlantWild) (CC BY-NC-ND 4.0: non-commercial use, no redistribution of modified images).
- **GVLiD** — *GrapeVine Leaf identification of the Diseases*. [Mendeley Data](https://data.mendeley.com/datasets/wkymf8bhcg) (CC BY 4.0).
- **ESCA-dataset** — Alessandrini, M. et al. [Mendeley Data](https://data.mendeley.com/datasets/89cnxc58kj/1) (CC BY 4.0).
- **Potato Leaf (Healthy and Late Blight)** — [Mendeley Data](https://data.mendeley.com/datasets/v4w72bsts5) (CC BY 4.0).
- **Plant Pathology 2021** — Thapa, R. et al., Cornell AgriTech. [Hugging Face](https://huggingface.co/datasets/timm/plant-pathology-2021) (Apache-2.0). The script downloads up to 500 photos per class, resized to 1024 px.
- **Tomato Leaf Disease Classification Dataset in Pakistan** — [Mendeley Data](https://data.mendeley.com/datasets/3mbnb82mxd) (CC BY 4.0). The script uses only the 830 original photos, without the augmented copies.
- **Tomato leaf diseases** — Daffodil International University. [Mendeley Data](https://data.mendeley.com/datasets/93h9p62kg4) (CC BY 4.0). Detached leaves photographed on a light background; the script downloads only the classes that exist in PlantVillage.

## 🏋️ Training

```bash
uv run python scripts/train.py
```

To write the model to another folder and compare it before replacing the current one:

```bash
uv run python scripts/train.py --output-dir models_new
```

> **GPU:** on Windows and Linux, `uv sync` already installs PyTorch with CUDA, and training uses an NVIDIA GPU automatically (on the CPU, training takes many hours). On Linux with an AMD GPU, use the Docker setup below. With a GPU, a full training run takes a few hours; keeping the `data/` folder on an SSD makes epochs much faster than on a hard drive.

Options:

- `--output-dir folder/` — writes the artifacts to another folder (the default `models/` replaces the model used by the app).
- `--epochs-stage1`, `--epochs-stage2`, `--patience` — maximum epochs of each stage (default 5 and 25) and epochs without improvement before stopping (default 5).
- `--unfreeze-blocks 9` and `--stage2-lr 1e-4` — how many backbone blocks to train in stage 2 and with which learning rate (default: the whole network; with field photos, this raised field accuracy from 72.7% to 78.3% compared with training only the last 3 blocks at $10^{-5}$).
- `--background-prob 0.5` — replaces the background of part of the PlantVillage leaves using the masks from the PlantVillage `segmented/` folder (place it at `data/raw/PlantVillage/segmented/`). Off by default: it made the model much more robust to artificial backgrounds, but with field photos in training it did not improve field accuracy and makes training ~3× slower.
- `--field-share 0.4` — fraction of each epoch drawn from field photos; `--no-field` ignores the field photos; `--field-datasets PlantDoc GVLiD ...` uses only the listed field datasets.
- `--use-processor` — also applies the `ImageProcessor` during training (see the note in the pipeline section).

How training works:

- **Split by leaf:** copies, rotations and flips of the same photo always stay in the same set (train, validation or test), so the test set never contains images already seen in training.
- **Checkpoint selection:** when field photos are present, the best model is chosen by accuracy on the validation field photos.
- **Consistent preprocessing:** training writes `preprocessing.json` next to the model, and the app applies exactly the same preprocessing at inference. Using different preprocessing at inference than in training drops accuracy (with the old checkpoint, enabling the `ImageProcessor` only at inference went from 99% to 64% on PlantVillage leaves). Checkpoints without this file keep the original behavior (with `ImageProcessor`).

Artifacts are written to `models/`:

- `best_model.pt` — best checkpoint (by validation accuracy, or field validation accuracy when field photos are present)
- `classes.json` — class index mapping
- `preprocessing.json` — whether the `ImageProcessor` was used in training
- `split.json` — train/val/test paths
- `history.json` — per-epoch metrics
- `metrics.json` — final test metrics (includes `field_test_accuracy` when field photos are present)
- `classification_report.txt` / `classification_report_campo.txt` — per-class precision/recall/F1
- `confusion_matrix.png` / `confusion_matrix_campo.png`

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
docker --context default compose run --rm train --batch-size 64 --output-dir models_new
```

Download the field datasets on the host (`uv run python scripts/download_field_datasets.py`) before training: two of them come as `.rar` files, and extracting them requires `bsdtar`, `unrar` or `7z`, which are not in the Docker image. On Linux, `bsdtar` comes with the `libarchive-tools` package.

## 🔍 Inference (CLI)

```bash
uv run python scripts/predict.py path/to/leaf.jpg --top 3
```

## 🧪 Evaluating a Folder of Photos

To measure the model in `models/` on many photos at once, organize them in a folder with one subfolder per class (named exactly after the class, e.g. `Tomato___Late_blight/`) and run:

```bash
uv run python scripts/evaluate_folder.py path/to/folder
```

The script uses the same inference path as the app, saves a spreadsheet (`resultado.csv`) with each photo's prediction and prints the overall accuracy, accuracy by confidence band, accuracy per class and the most common confusions.

## 📦 Publishing the Model (GitHub Release)

The trained model is not stored in the repository: it is published as a release asset, and the app downloads it on its own. After training the final model:

```bash
uv run python scripts/package_model.py
```

This creates `dist/leafguardai-model.zip`, with `best_model.pt`, `classes.json`, `preprocessing.json` and `metrics.json` (use `--models-dir` to package another folder). On GitHub, create a release (e.g. `v1.0.0`) and attach this file **with exactly this name**. The app always downloads the model from the latest release.

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

## 📄 License

The source code is licensed under the [GNU GPL 3.0](LICENSE). The trained model weights published in the releases are a separate file and are not covered by it: since they were trained on third-party datasets, including PlantWild (CC BY-NC-ND 4.0), they are provided **for non-commercial use only**. The license of each dataset is listed in the [Dataset](#-dataset) section.
