import cv2
import numpy as np
from utils import non_max_suppression_fast

class TemplateMatcher:
    def __init__(self, image, template, params):
        self.image = image
        self.template = template
        self.params = params

    def detect(self):
        threshold = self.params["threshold"]
        rot_inv = self.params["rotation_invariant"]
        multi_scale = self.params["multi_scale"]
        scale_r = self.params["scale_range"]
        methods = self.params["methods"]

        detected_points = []
        method = cv2.TM_CCOEFF_NORMED

        # Optimize scales - reduce number of scale steps for performance
        scales = [1.0]
        if multi_scale:
            # Reduced step size and limited range for better performance
            step = 0.1 
            scale_min = max(0.3, 1 - scale_r) 
            scale_max = min(1.7, 1 + scale_r)  
            scales = np.arange(scale_min, scale_max + step, step)
            # Limit to maximum 8 scales for performance
            if len(scales) > 8:
                scales = np.linspace(scale_min, scale_max, 8)

        # Pre-process images once
        if methods["grayscale"]:
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if methods["edge"]:
            img_edge = cv2.Canny(self.image, 100, 200)

        def rotated_templates(template):
            angles = range(0, 360, 30) if rot_inv else [0]
            if multi_scale and len(scales) > 4:
                angles = range(0, 360, 45) 
            
            for angle in angles:
                if angle == 0:
                    yield template
                else:
                    M = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), angle, 1)
                    rotated = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))
                    yield rotated

        # Process each method
        active_methods = [name for name, use in methods.items() if use]
        
        for method_name in active_methods:
            if method_name == "grayscale":
                base_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
                search_img = img_gray
            elif method_name == "color":
                base_template = self.template
                search_img = self.image
            elif method_name == "edge":
                base_template = cv2.Canny(self.template, 100, 200)
                search_img = img_edge
            else:
                continue

            # Skip if template is too small after edge detection
            if method_name == "edge" and (base_template.shape[0] < 10 or base_template.shape[1] < 10):
                continue

            for scale in scales:
                if scale != 1.0:
                    new_w = max(10, int(base_template.shape[1] * scale))  # Minimum size check
                    new_h = max(10, int(base_template.shape[0] * scale))
                    # Skip if scaled template is too large or too small
                    if new_w > search_img.shape[1] * 0.8 or new_h > search_img.shape[0] * 0.8:
                        continue
                    if new_w < 5 or new_h < 5:
                        continue
                    scaled_template = cv2.resize(base_template, (new_w, new_h))
                else:
                    scaled_template = base_template

                # Skip if template is larger than search image
                if scaled_template.shape[0] >= search_img.shape[0] or scaled_template.shape[1] >= search_img.shape[1]:
                    continue

                templates = rotated_templates(scaled_template)

                for tmpl in templates:
                    # Skip if rotated template is too large
                    if tmpl.shape[0] >= search_img.shape[0] or tmpl.shape[1] >= search_img.shape[1]:
                        continue
                    
                    try:
                        res = cv2.matchTemplate(search_img, tmpl, method)
                        loc = np.where(res >= threshold)
                        for pt in zip(*loc[::-1]):
                            detected_points.append((pt, tmpl.shape[::-1]))
                    except cv2.error:
                        # Skip if template matching fails
                        continue

        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        
        # Apply non-maximum suppression to remove overlapping detections
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), 0.3)
        else:
            boxes = []

        # Draw rectangles on a copy of the image
        detected_img = self.image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        return boxes, detected_img


class ColorSegmentationMatcher:
    def __init__(self, image, template, params):
        self.image = image
        self.template = template
        self.params = params

    def detect(self):
        color_tolerance = self.params["color_tolerance"]
        min_area = self.params["min_area"]
        erosion_iterations = self.params["erosion_iterations"]
        dilation_iterations = self.params["dilation_iterations"]
        
        # Get dominant color from the template ROI (use median for robustness)
        template_bgr = self.template.copy()
        
        # Calculate color statistics in BGR space (simpler and more reliable)
        b_med = np.median(template_bgr[:, :, 0])
        g_med = np.median(template_bgr[:, :, 1])
        r_med = np.median(template_bgr[:, :, 2])
        
        # Create color range in BGR space
        tolerance_bgr = color_tolerance * 3  # Scale tolerance for BGR
        lower_bgr = np.array([
            max(0, b_med - tolerance_bgr),
            max(0, g_med - tolerance_bgr),
            max(0, r_med - tolerance_bgr)
        ])
        upper_bgr = np.array([
            min(255, b_med + tolerance_bgr),
            min(255, g_med + tolerance_bgr),
            min(255, r_med + tolerance_bgr)
        ])
        
        # Create mask in BGR color space
        mask = cv2.inRange(self.image, lower_bgr, upper_bgr)
        
        # Also try HSV approach for better color matching
        template_hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Get HSV statistics
        h_med = np.median(template_hsv[:, :, 0])
        s_med = np.median(template_hsv[:, :, 1])
        v_med = np.median(template_hsv[:, :, 2])
        
        # HSV ranges (more conservative)
        h_tol = min(20, color_tolerance)  # Limit hue tolerance
        s_tol = color_tolerance * 2
        v_tol = color_tolerance * 2
        
        lower_hsv = np.array([
            max(0, h_med - h_tol),
            max(30, s_med - s_tol),  # Avoid very low saturation (grayish colors)
            max(30, v_med - v_tol)   # Avoid very dark colors
        ])
        upper_hsv = np.array([
            min(179, h_med + h_tol),
            min(255, s_med + s_tol),
            min(255, v_med + v_tol)
        ])
        
        # Create HSV mask
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        
        # Combine both masks (BGR and HSV) for better results
        mask_combined = cv2.bitwise_or(mask, mask_hsv)
        
        # Apply morphological operations to clean up the mask
        if erosion_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_combined = cv2.erode(mask_combined, kernel, iterations=erosion_iterations)
        
        if dilation_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_combined = cv2.dilate(mask_combined, kernel, iterations=dilation_iterations)
        
        # Find contours
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours and create bounding boxes
        boxes = []
        detected_img = self.image.copy()
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very thin or very wide rectangles (likely noise)
                aspect_ratio = w / h
                if 0.2 <= aspect_ratio <= 5.0:  # Reasonable aspect ratio
                    boxes.append([x, y, x + w, y + h])
        
        # Apply non-maximum suppression
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), 0.4)
            
            # Draw results
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        return boxes, detected_img
