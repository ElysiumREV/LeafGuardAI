import cv2
import numpy as np

size = (256, 256)
segment_leaf = True

def load_image(path: str):

    # Image Reading
    bgr_image = cv2.imread(path, 1)


    # Leaf Segmentation
    leaf_mask = None

    if segment_leaf:
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        lower = np.array([15, 30, 30])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv_image, lower, upper)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = 10
            x0 = max(x - margin, 0)
            y0 = max(y - margin, 0)
            x1 = min(x + w + margin, bgr_image.shape[1])
            y1 = min(y + h + margin, bgr_image.shape[0])
            bgr_image = bgr_image[y0:y1, x0:x1]
            leaf_mask = mask[y0:y1, x0:x1]

    #Color correction
    float_image = bgr_image.astype(np.float32)

    if leaf_mask is not None and leaf_mask.any() and (~(leaf_mask > 0)).any():
        background_pixels = float_image[leaf_mask == 0]
        mean_b, mean_g, mean_r = background_pixels[:, 0].mean(), background_pixels[:, 1].mean(), background_pixels[:, 2].mean()
    else:
        mean_b, mean_g, mean_r = [float_image[:, :, i].mean() for i in range(3)]

    mean_gray = (mean_b + mean_g + mean_r) / 3

    float_image[:, :, 0] *= mean_gray / max(mean_b, 1e-6)
    float_image[:, :, 1] *= mean_gray / max(mean_g, 1e-6)
    float_image[:, :, 2] *= mean_gray / max(mean_r, 1e-6)

    bgr_image = np.clip(float_image, 0, 255).astype(np.uint8)

    #Denoising
    bgr_image = cv2.fastNlMeansDenoisingColored(bgr_image, None, 1, 1, 7, 21)

    #Contrast
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    bgr_image = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)

    #Light
    strength = 0.6
    radius = 2
    blurred = cv2.GaussianBlur(bgr_image, (0,0), radius)
    bgr_image = cv2.addWeighted(bgr_image, 1 + strength, blurred, -strength, 0)
    
    #Resize
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)

    # print(denoising_value_test(bgr_image))

    return bgr_image

# def denoising_value_test(bgr_image):
#     gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()