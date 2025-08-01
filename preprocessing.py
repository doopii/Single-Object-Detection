import cv2
import numpy as np

def resize_image(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def apply_image_preprocessing(img, brightness=0, contrast=1.0, blur=0, sharpen=False, denoise=False):
    """Apply all preprocessing steps to an image"""
    processed = img.astype(np.float32)
    
    # Brightness adjustment
    if brightness != 0:
        processed = cv2.add(processed, np.ones(processed.shape, dtype=np.float32) * brightness)
    
    # Contrast adjustment
    if contrast != 1.0:
        processed = cv2.multiply(processed, contrast)
    
    # Clip values to valid range
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    # Gaussian blur
    if blur > 0:
        ksize = blur * 2 + 1  
        processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
    
    # Denoising
    if denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    # Sharpening
    if sharpen:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    return processed
