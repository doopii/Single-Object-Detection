import cv2
import numpy as np

print("Detection module loaded")

# --- Preprocessing Functions ---
def preprocess_image(image_file):
    """Preprocess the image for detection (convert to grayscale, apply blur, etc.)."""
    # Convert uploaded file to an image
    img_bytes = image_file.read()

    # Check if the byte stream is empty
    if len(img_bytes) == 0:
        print("Error: Image byte stream is empty.")
        return None, None, None
    
    # Debugging: Print length of img_bytes
    print(f"Image bytes length: {len(img_bytes)}")

    # Try decoding the image
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Check if the image is successfully decoded
    if img is None:
        print("Error: Image could not be processed (invalid format).")
        return None, None, None
    
    # Convert to grayscale and apply Gaussian Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    print("Image preprocessing completed.")
    return img, gray, blurred

def apply_histogram_equalization(gray_img):
    """Apply histogram equalization to enhance contrast."""
    print("Applying histogram equalization...")
    return cv2.equalizeHist(gray_img)

def apply_watershed(img):
    """Apply Watershed algorithm if needed for separating touching objects."""
    print("Applying Watershed algorithm...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_bg = cv2.dilate(thresh, None, iterations=3)
    
    sure_fg = np.uint8(sure_fg)
    sure_bg = np.uint8(sure_bg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    cv2.watershed(img, markers)
    img[markers == -1] = [0, 0, 255]
    
    print("Watershed algorithm applied.")
    return img

# --- Object Detection Functions ---
def detect_keypoints_and_matches(roi, full_image):
    """Detect ORB keypoints and match features between ROI and full image."""
    print("Detecting keypoints and matching...")
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors for ROI and full image
    keypoints_roi, descriptors_roi = orb.detectAndCompute(roi, None)
    keypoints_img, descriptors_img = orb.detectAndCompute(full_image, None)
    
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_roi, descriptors_img)
    
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Found {len(matches)} matches.")
    return keypoints_roi, keypoints_img, matches

def draw_bounding_boxes(image, keypoints_img, matches):
    """Draw bounding boxes around matched areas in the image."""
    img_with_boxes = image.copy()
    img_with_boxes = cv2.drawMatches(image, keypoints_img, image, keypoints_img, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print("Bounding boxes drawn.")
    return img_with_boxes

def filter_matches(matches, min_distance=30):
    """Filter matches based on score (distance)."""
    filtered_matches = [match for match in matches if match.distance < min_distance]
    print(f"Filtered {len(filtered_matches)} matches.")
    return filtered_matches

def save_image(image, filename):
    """Save the result image."""
    cv2.imwrite(filename, image)
    print(f"Result image saved as {filename}")

def generate_report(keypoints_count, matches_count, processing_time, roi_size):
    """Generate a simple report with details about the detection process."""
    report = f"Detection Report:\n"
    report += f"- Number of keypoints: {keypoints_count}\n"
    report += f"- Number of matches: {matches_count}\n"
    report += f"- Processing Time: {processing_time:.2f} seconds\n"
    report += f"- Selected ROI Size: {roi_size} pixels"
    
    print("Report generated:")
    print(report)
    return report
