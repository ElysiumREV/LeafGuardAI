import cv2

def load_image(path: str):
    image = cv2.imread(path, 1)
    return image