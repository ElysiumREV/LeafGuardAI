import cv2
import numpy as np
from os import PathLike
from PIL import Image
from typing import Union, Tuple, Optional

class ImageProcessor:
    """
    Unified image processing pipeline for LeafGuardAI.
    Handles segmentation, color correction, denoising, and contrast enhancement
    to ensure consistency between training and inference.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        segment_leaf: bool = True,
        color_correction: bool = True,
        denoise: bool = True,
        contrast: bool = True,
        sharpen: bool = True
    ):
        self.target_size = target_size
        self.segment_leaf = segment_leaf
        self.color_correction = color_correction
        self.denoise = denoise
        self.contrast = contrast
        self.sharpen = sharpen

    def process(self, image_source: Union[str, PathLike, np.ndarray]) -> Image.Image:
        """
        Main entry point for processing.
        Args:
            image_source: Path to image file or a BGR numpy array.
        Returns:
            A processed PIL Image in RGB format.
        """
        # 1. Loading
        if isinstance(image_source, (str, PathLike)):
            image = cv2.imread(str(image_source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image at path: {image_source}")
        else:
            image = image_source.copy()

        # 2. Processing Pipeline
        mask = None
        if self.segment_leaf:
            image, mask = self._segment_leaf(image)

        if self.color_correction:
            image = self._correct_color(image, mask)

        if self.denoise:
            image = self._denoise(image)

        if self.contrast:
            image = self._enhance_contrast(image)

        if self.sharpen:
            image = self._sharpen(image)

        image = self._resize(image)

        # 3. Convert BGR NumPy -> RGB PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    def _segment_leaf(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Isolate the leaf from the background using HSV masking."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green range mask
        lower = np.array([15, 30, 30])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = 10
            x0 = max(x - margin, 0)
            y0 = max(y - margin, 0)
            x1 = min(x + w + margin, image.shape[1])
            y1 = min(y + h + margin, image.shape[0])

            cropped_image = image[y0:y1, x0:x1]
            cropped_mask = mask[y0:y1, x0:x1]
            return cropped_image, cropped_mask

        return image, None

    def _correct_color(self, image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        """Remove color casts using background-aware grayscale normalization."""
        float_image = image.astype(np.float32)

        if mask is not None and mask.any() and (~(mask > 0)).any():
            # Use background pixels for mean calculation
            background_pixels = float_image[mask == 0]
            mean_b, mean_g, mean_r = background_pixels[:, 0].mean(), background_pixels[:, 1].mean(), background_pixels[:, 2].mean()
        else:
            # Fallback to global mean
            mean_b, mean_g, mean_r = [float_image[:, :, i].mean() for i in range(3)]

        mean_gray = (mean_b + mean_g + mean_r) / 3

        # Normalize channels
        float_image[:, :, 0] *= mean_gray / max(mean_b, 1e-6)
        float_image[:, :, 1] *= mean_gray / max(mean_g, 1e-6)
        float_image[:, :, 2] *= mean_gray / max(mean_r, 1e-6)

        return np.clip(float_image, 0, 255).astype(np.uint8)

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using Fast Non-Local Means Denoising."""
        return cv2.fastNlMeansDenoisingColored(image, None, 1, 1, 7, 21)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE in LAB color space to enhance local contrast."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Perform unsharp masking to enhance edges."""
        strength = 0.6
        radius = 2
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
