#!/usr/bin/env python3
"""
Test script for SIFT detection functionality
"""

import cv2
import numpy as np
from detection import SIFTMatcher

def test_sift_detection():
    # Create a simple test image with some texture
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Add some textured content
    cv2.putText(image, "SIFT TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.rectangle(image, (200, 150), (400, 300), (0, 255, 0), 2)
    cv2.circle(image, (300, 225), 50, (255, 0, 0), -1)
    
    # Create a template (part of the image)
    template = image[80:120, 40:200]  # Extract "SIFT" text region
    
    # Test SIFT detection parameters
    params = {
        "min_matches": 4,
        "match_ratio": 0.7,
        "ransac_threshold": 5.0
    }
    
    print("Testing SIFT detection...")
    try:
        matcher = SIFTMatcher(image, template, params)
        boxes, result_img = matcher.detect()
        
        print(f"SIFT detection completed!")
        print(f"Number of detections: {len(boxes)}")
        if len(boxes) > 0:
            print(f"Detection boxes: {boxes}")
        
        # Save result image
        cv2.imwrite("sift_test_result.png", result_img)
        print("Result saved as 'sift_test_result.png'")
        
        return True
        
    except Exception as e:
        print(f"SIFT detection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sift_detection()
    if success:
        print("✅ SIFT implementation test passed!")
    else:
        print("❌ SIFT implementation test failed!")
