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

        # Configure scale parameters for multi-scale detection
        scales = [1.0]
        if multi_scale:
            step = 0.1 
            scale_min = max(0.3, 1 - scale_r) 
            scale_max = min(1.7, 1 + scale_r)  
            scales = np.arange(scale_min, scale_max + step, step)
            # Limit scales for performance optimization
            if len(scales) > 8:
                scales = np.linspace(scale_min, scale_max, 8)

        # Pre-process images for different detection methods
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

        # Process each enabled detection method
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

            # Multi-scale template matching
            for scale in scales:
                if scale != 1.0:
                    new_w = max(10, int(base_template.shape[1] * scale))
                    new_h = max(10, int(base_template.shape[0] * scale))
                    # Skip invalid template sizes
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

                # Generate rotated templates and perform matching
                templates = rotated_templates(scaled_template)

                for tmpl in templates:
                    # Skip oversized rotated templates
                    if tmpl.shape[0] >= search_img.shape[0] or tmpl.shape[1] >= search_img.shape[1]:
                        continue
                    
                    try:
                        res = cv2.matchTemplate(search_img, tmpl, method)
                        loc = np.where(res >= threshold)
                        for pt in zip(*loc[::-1]):
                            detected_points.append((pt, tmpl.shape[::-1]))
                    except cv2.error:
                        continue

        # Convert detection points to bounding boxes
        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        
        # Apply non-maximum suppression and draw results
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), 0.3)
        else:
            boxes = []

        detected_img = self.image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        return boxes, detected_img


class ColorSegmentationMatcher:
    def __init__(self, image, template, params):
        self.image = image
        self.template = template
        self.params = params

    def apply_watershed(self, mask):
        
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # Distance transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Find seed points for watershed (object centers)
        max_val = dist_transform.max()
        threshold_val = max(3, 0.4 * max_val)
        local_maxima = dist_transform > threshold_val
        
        # Clean up noise in seed detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        local_maxima = cv2.morphologyEx(local_maxima.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Create watershed markers
        num_labels, markers = cv2.connectedComponents(local_maxima)

        # Skip watershed if only 1 object found (bg + 2 labels)
        if num_labels < 3:  
            return mask

        # Add background marker (bg = 0, fg = 1)
        markers = markers + 1
        markers[mask == 0] = 0

        # Apply watershed
        if len(self.image.shape) == 3:
            watershed_img = cv2.watershed(self.image, markers.copy())
        else:
            temp_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            watershed_img = cv2.watershed(temp_img, markers.copy())
        
        # Create new mask from watershed result
        # watershed_img == -1 are the watershed lines 
        # watershed_img > 1 are the separated objects
        watershed_mask = (watershed_img > 1).astype(np.uint8) * 255
        
        return watershed_mask

    def detect(self):
        color_tolerance = self.params["color_tolerance"]
        min_area = self.params["min_area"]
        erosion_iterations = self.params["erosion_iterations"]
        dilation_iterations = self.params["dilation_iterations"]
        use_watershed = self.params.get("use_watershed", False)
        
        # Extract dominant color using median
        template_bgr = self.template.copy()
        
        b_med = np.median(template_bgr[:, :, 0])
        g_med = np.median(template_bgr[:, :, 1])
        r_med = np.median(template_bgr[:, :, 2])
        
        # Create BGR color range
        tolerance_bgr = color_tolerance * 3
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
        
        # Create BGR mask
        mask = cv2.inRange(self.image, lower_bgr, upper_bgr)
        
        # Generate HSV mask for better color matching
        template_hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        h_med = np.median(template_hsv[:, :, 0])
        s_med = np.median(template_hsv[:, :, 1])
        v_med = np.median(template_hsv[:, :, 2])
        
        # HSV ranges with conservative tolerances
        h_tol = min(20, color_tolerance)
        s_tol = color_tolerance * 2
        v_tol = color_tolerance * 2
        
        lower_hsv = np.array([
            max(0, h_med - h_tol),
            max(30, s_med - s_tol),  
            max(30, v_med - v_tol)   
        ])
        upper_hsv = np.array([
            min(179, h_med + h_tol),
            min(255, s_med + s_tol),
            min(255, v_med + v_tol)
        ])
        
        mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        
        # Combine BGR and HSV masks
        mask_combined = cv2.bitwise_or(mask, mask_hsv)
        
        # Apply morphological operations to clean mask
        if erosion_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_combined = cv2.erode(mask_combined, kernel, iterations=erosion_iterations)
        
        if dilation_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_combined = cv2.dilate(mask_combined, kernel, iterations=dilation_iterations)
        
        # Apply watershed segmentation if enabled
        if use_watershed:
            # Apply closing operation before watershed to fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask_before_watershed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
            mask_combined = self.apply_watershed(mask_before_watershed)
        
        # Find and filter contours
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        detected_img = self.image.copy()
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 <= aspect_ratio <= 5.0:
                    boxes.append([x, y, x + w, y + h])
        
        # Apply non-maximum suppression and draw results
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), 0.4)
            
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        return boxes, detected_img
