import cv2
import numpy as np

def resize_image(img, max_dim):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))

def apply_image_preprocessing(img, brightness=0, contrast=1.0, blur=0, blur_type="gaussian", sharpen=False, denoise=False):

    processed = img.astype(np.float32)
    
    # Brightness adjustment
    if brightness != 0:
        processed = cv2.add(processed, np.ones(processed.shape, dtype=np.float32) * brightness)
    
    # Contrast adjustment
    if contrast != 1.0:
        processed = cv2.multiply(processed, contrast)
    
    # Clip values to valid range
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    # Apply blur based on selected type
    if blur > 0:
        ksize = blur * 2 + 1
        if blur_type.lower() == "gaussian":
            # Gaussian blur - fast, smooth, natural blur good for general noise reduction
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
        elif blur_type.lower() == "median":
            # Median blur - excellent for salt-and-pepper noise, preserves edges better
            processed = cv2.medianBlur(processed, ksize)
        else:
            # Default to Gaussian if invalid type provided
            processed = cv2.GaussianBlur(processed, (ksize, ksize), 0)
    
    # Non-Local Means denoising (slow but high quality)
    if denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
    
    # Sharpening (high-pass filter)
    if sharpen:
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    return processed
