import cv2
import numpy as np
import time
from detection import preprocess_image, apply_histogram_equalization, apply_watershed, detect_keypoints_and_matches, draw_bounding_boxes, save_image, generate_report, filter_matches

# Path to your image for testing
image_path = 'screwDrivers.jpg'  # Update with your test image path

# --- Step-by-Step Testing ---
start_time = time.time()

# Check if the file exists
import os
if not os.path.isfile(image_path):
    print(f"Error: The image file at {image_path} was not found.")
else:
    # Step 1: Image Preprocessing
    with open(image_path, 'rb') as f:
        img, gray, blurred = preprocess_image(f)
    
    if img is None:
        print("Error: Image could not be processed.")
    else:
        # Step 2: Apply Histogram Equalization (Optional)
        gray = apply_histogram_equalization(gray)

        # Step 3: Apply Watershed (Optional)
        img_processed = apply_watershed(img)

        # Step 4: Object Detection (ORB Keypoint Detection and Matching)
        keypoints_roi, keypoints_img, matches = detect_keypoints_and_matches(gray, img_processed)

        # Debugging: Check the number of keypoints and matches
        print(f"Number of keypoints in ROI: {len(keypoints_roi)}")
        print(f"Number of keypoints in full image: {len(keypoints_img)}")
        print(f"Number of matches found: {len(matches)}")

        if len(keypoints_img) > 0 and len(matches) > 0:
            # Check if matches are valid
            print(f"Number of valid matches: {len(matches)}")
            
            # Step 5: Draw Bounding Boxes on Matches
            img_with_boxes = draw_bounding_boxes(img_processed, keypoints_img, matches)

            # Display the result
            cv2.imshow("Detected Objects", img_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Step 6: Filter Matches
            filtered_matches = filter_matches(matches, min_distance=30)

            # Step 7: Generate Report
            processing_time = time.time() - start_time
            roi_size = img_with_boxes.shape[0] * img_with_boxes.shape[1]  # Example ROI size (image size in pixels)
            report = generate_report(len(keypoints_img), len(filtered_matches), processing_time, roi_size)

            # Show the report
            print(report)

            # Step 8: Save the Result Image
            save_image(img_with_boxes, "detection_result.jpg")
        else:
            print("Error: No valid keypoints or matches found.")
