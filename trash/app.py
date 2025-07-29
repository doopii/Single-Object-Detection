import streamlit as st
import cv2
import numpy as np
import time
from detection import preprocess_image, apply_histogram_equalization, apply_watershed, detect_keypoints_and_matches, draw_bounding_boxes, save_image, generate_report
from streamlit_drawable_canvas import st_canvas

# Streamlit title
st.title("Object Detection System")

# --- Image Upload ---
image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if image_file:
    # Debugging: Print file type and size
    st.write(f"File type: {image_file.type}")
    st.write(f"File size: {image_file.size} bytes")
    
    # Convert uploaded file to an image
    img_bytes = image_file.read()
    if len(img_bytes) == 0:
        st.error("Uploaded file is empty or invalid.")
    else:
        # Process image if not empty
        img, gray, blurred = preprocess_image(image_file)

# User Inputs for optional processing steps
apply_hist_eq = st.checkbox("Apply Histogram Equalization", value=False)
apply_watershed = st.checkbox("Apply Watershed Algorithm", value=False)

# Threshold sliders for object detection
st.write("Select Threshold for Object Detection (Brightness/Intensity):")
low_thresh = st.slider("Low Threshold", min_value=0, max_value=255, value=50)
high_thresh = st.slider("High Threshold", min_value=0, max_value=255, value=150)

if image_file:
    # Convert uploaded file to an image
    img_bytes = image_file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is not None:
        # Display the original image
        st.image(img, channels="BGR", caption="Original Image", use_column_width=True)

        # --- ROI Selection via Region Buttons ---
        st.write("Select Region of Interest (ROI) for Detection:")
        region = st.radio("Choose Region", ["Top Left", "Top Right", "Bottom Left", "Bottom Right"])

        # Define region boundaries
        if region == "Top Left":
            roi = img[:img.shape[0]//2, :img.shape[1]//2]
        elif region == "Top Right":
            roi = img[:img.shape[0]//2, img.shape[1]//2:]
        elif region == "Bottom Left":
            roi = img[img.shape[0]//2:, :img.shape[1]//2]
        else:
            roi = img[img.shape[0]//2:, img.shape[1]//2:]

        st.image(roi, caption=f"Selected {region} ROI", use_column_width=True)

        # --- Threshold-based Object Detection ---
        st.write("Detecting Objects Based on Intensity Threshold...")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray, low_thresh, high_thresh, cv2.THRESH_BINARY)
        st.image(thresh_img, caption="Thresholded Image", use_column_width=True)

        # Step 1: Start timer and begin image preprocessing
        start_time = time.time()  # Record start time here

        # Step 2: Image Preprocessing
        gray, blurred = preprocess_image(image_file)
        if apply_hist_eq:
            gray = apply_histogram_equalization(gray)

        # Step 3: Apply Watershed if selected
        if apply_watershed:
            img_processed = apply_watershed(img)
        else:
            img_processed = img

        # --- Object Detection ---
        st.write("Detecting Keypoints and Matches...")
        keypoints_roi, keypoints_img, matches = detect_keypoints_and_matches(gray, img_processed)

        if keypoints_img is not None:
            # Step 4: Draw Bounding Boxes on Matches
            img_with_boxes = draw_bounding_boxes(img_processed, keypoints_img, matches)

            # Display the result
            st.image(img_with_boxes, caption="Processed Image", channels="BGR", use_column_width=True)

            # Generate Report
            processing_time = time.time() - start_time
            roi_size = img_with_boxes.shape[0] * img_with_boxes.shape[1]  # Example ROI size (image size in pixels)
            report = generate_report(len(keypoints_img), len(matches), processing_time, roi_size)

            # Show the report
            st.text_area("Detection Report", value=report, height=200)

            # Save the result image
            result_image_path = "detection_result.jpg"
            save_image(img_with_boxes, result_image_path)

            st.download_button(
                label="Download Processed Image",
                data=open(result_image_path, "rb").read(),
                file_name="detection_result.jpg",
                mime="image/jpeg"
            )
        else:
            st.error("No keypoints detected. Please check the image quality.")
