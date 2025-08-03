#!/usr/bin/env python3
"""
SIFT Parameter Testing Script - Find optimal settings for your images
"""

import cv2
import numpy as np
from detection import SIFTMatcher
import os

def test_sift_parameters(image_path, template_region=None):
    """
    Test different SIFT parameter combinations
    
    Args:
        image_path: Path to your test image
        template_region: (x, y, w, h) - region to use as template, or None for manual selection
    """
    
    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"🖼️  Loaded image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Get template region
    if template_region is None:
        print("\n📋 Select template region:")
        print("   1. Click and drag to select the object you want to detect")
        print("   2. Press any key to confirm, ESC to cancel")
        
        clone = image.copy()
        roi_pts = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_pts.clear()
                roi_pts.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                roi_pts.append((x, y))
                cv2.rectangle(clone, roi_pts[0], roi_pts[1], (0, 255, 0), 2)
                cv2.imshow("Select Template", clone)
        
        cv2.namedWindow("Select Template")
        cv2.setMouseCallback("Select Template", mouse_callback)
        cv2.imshow("Select Template", clone)
        
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == 27 or len(roi_pts) != 2:  # ESC pressed or invalid selection
            print("❌ Template selection cancelled")
            return
        
        x1, y1 = roi_pts[0]
        x2, y2 = roi_pts[1]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        template_region = (x1, y1, x2-x1, y2-y1)
    
    x, y, w, h = template_region
    template = image[y:y+h, x:x+w]
    
    print(f"📐 Template size: {w}x{h} pixels")
    
    # Test different parameter combinations
    test_configs = [
        # (min_matches, match_ratio, ransac_threshold, description)
        (4,  0.7, 5.0, "Lenient - Good for simple objects"),
        (10, 0.7, 5.0, "Balanced - Default settings"),
        (15, 0.6, 3.0, "Strict - High precision"),
        (8,  0.8, 7.0, "Tolerant - More detections"),
        (20, 0.5, 2.0, "Very Strict - Minimal false positives"),
        (6,  0.75, 8.0, "Flexible - Various perspectives")
    ]
    
    print("\n🧪 Testing SIFT parameter combinations:")
    print("=" * 60)
    
    best_config = None
    best_count = 0
    
    for i, (min_matches, ratio, ransac, desc) in enumerate(test_configs, 1):
        params = {
            "min_matches": min_matches,
            "match_ratio": ratio,
            "ransac_threshold": ransac
        }
        
        try:
            matcher = SIFTMatcher(image, template, params)
            boxes, result_img = matcher.detect()
            
            print(f"{i}. {desc}")
            print(f"   Parameters: min_matches={min_matches}, ratio={ratio:.2f}, ransac={ransac:.1f}")
            print(f"   Results: {len(boxes)} detection(s)")
            
            if len(boxes) > best_count:
                best_count = len(boxes)
                best_config = (min_matches, ratio, ransac, desc)
            
            # Save result
            output_path = f"sift_test_{i:02d}.png"
            cv2.imwrite(output_path, result_img)
            print(f"   Saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    if best_config:
        min_matches, ratio, ransac, desc = best_config
        print(f"🏆 Best configuration: {desc}")
        print(f"   Parameters: min_matches={min_matches}, ratio={ratio:.2f}, ransac={ransac:.1f}")
        print(f"   Detections: {best_count}")
        print(f"\n💡 Recommended UI settings:")
        print(f"   - Minimum Feature Matches: {min_matches}")
        print(f"   - Match Quality Ratio: {ratio:.2f}")
        print(f"   - RANSAC Threshold: {ransac:.1f}")
    else:
        print("❌ No successful detections found")
        print("\n💡 Try these troubleshooting steps:")
        print("   1. Use an image with more texture/details")
        print("   2. Select a larger template region")
        print("   3. Ensure good contrast and sharpness")
    
    return best_config

def analyze_image_sift_suitability(image_path):
    """Analyze if an image is suitable for SIFT detection"""
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot load image: {image_path}")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect SIFT features in the whole image
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Analyze image properties
    total_pixels = gray.shape[0] * gray.shape[1]
    feature_density = len(keypoints) / total_pixels * 10000  # features per 10k pixels
    
    # Calculate image metrics
    blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()  # Higher = sharper
    contrast = gray.std()  # Higher = more contrast
    
    print(f"\n📊 SIFT Suitability Analysis:")
    print(f"Image size: {gray.shape[1]}x{gray.shape[0]} pixels")
    print(f"SIFT keypoints found: {len(keypoints)}")
    print(f"Feature density: {feature_density:.1f} features per 10k pixels")
    print(f"Sharpness score: {blur_metric:.1f} (>100 is good)")
    print(f"Contrast score: {contrast:.1f} (>40 is good)")
    
    # Provide recommendations
    print(f"\n💡 Recommendations:")
    if len(keypoints) > 100:
        print("✅ Excellent feature count - great for SIFT!")
    elif len(keypoints) > 50:
        print("⚠️  Moderate features - SIFT should work")
    else:
        print("❌ Few features - consider Template or Color Segmentation")
    
    if blur_metric > 100:
        print("✅ Good image sharpness")
    else:
        print("⚠️  Image may be blurry - consider sharpening")
    
    if contrast > 40:
        print("✅ Good contrast")
    else:
        print("⚠️  Low contrast - consider contrast adjustment")
    
    # Save visualization
    img_with_kp = cv2.drawKeypoints(image, keypoints, None, 
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("sift_features_visualization.png", img_with_kp)
    print(f"\n💾 Saved feature visualization: sift_features_visualization.png")

if __name__ == "__main__":
    print("🔍 SIFT Parameter Testing Tool")
    print("=" * 50)
    
    # Ask user for image path
    image_path = input("Enter image path (or press Enter for example): ").strip()
    
    if not image_path:
        # Create a sample image for testing
        print("📝 Creating sample textured image...")
        sample = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(sample, "OpenCV SIFT TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(sample, "Feature Detection", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.rectangle(sample, (200, 200), (400, 350), (0, 255, 0), 2)
        # Add some texture
        for i in range(0, 600, 20):
            cv2.line(sample, (i, 250), (i, 300), (100, 100, 100), 1)
        
        image_path = "sift_sample.png"
        cv2.imwrite(image_path, sample)
        print(f"📄 Created sample image: {image_path}")
    
    # Analyze image
    analyze_image_sift_suitability(image_path)
    
    # Test parameters
    print(f"\n🧪 Starting parameter testing...")
    best_config = test_sift_parameters(image_path)
    
    print(f"\n✅ Testing complete! Check the output images for results.")
