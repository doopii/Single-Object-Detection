import cv2
import numpy as np
from utils import non_max_suppression_fast

class TemplateMatcher:
    def __init__(self, image, template, params, mask=None):
        self.image = image
        self.template = template
        self.params = params
        self.mask = mask  

    def _generate_scales(self, multi_scale, scale_r):
        """Generate scale values for multi-scale matching"""
        if not multi_scale:
            return [1.0]
        
        # Make these configurable later
        step = self.params.get("scale_step", 0.1)
        scale_min = max(0.3, 1 - scale_r)
        scale_max = min(1.7, 1 + scale_r)
        max_scales = self.params.get("max_scales", 8)
        
        scales = np.arange(scale_min, scale_max + step, step)
        if len(scales) > max_scales:
            scales = np.linspace(scale_min, scale_max, max_scales)
        
        return scales

    def _get_rotation_angles(self, rot_inv, multi_scale, scales):
        """Get rotation angles based on settings"""
        if not rot_inv:
            return [0]
        
        # Use configurable rotation step
        base_step = self.params.get("rotation_step", 30)
        performance_step = self.params.get("rotation_step_multiscale", 45)
        
        step = performance_step if multi_scale and len(scales) > 4 else base_step
        return range(0, 360, step)

    def _apply_mask_to_template(self, template, method_type):
        """Apply mask to template based on method type"""
        if self.mask is None:
            return template
            
        if self.mask.shape[:2] != template.shape[:2]:
            # Resize mask to match template if needed
            mask_resized = cv2.resize(self.mask, (template.shape[1], template.shape[0]))
        else:
            mask_resized = self.mask
            
        masked_template = template.copy()
        if len(template.shape) == 3:  # Color template
            masked_template[mask_resized == 0] = [0, 0, 0]
        else:  # Grayscale template
            masked_template[mask_resized == 0] = 0
            
        return masked_template

    def _prepare_method_data(self, method_name):
        """Prepare template and search image for specific method"""
        canny_low = self.params.get("canny_low", 100)
        canny_high = self.params.get("canny_high", 200)
        
        if method_name == "grayscale":
            template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
            search_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif method_name == "color":
            template = self.template
            search_img = self.image
        elif method_name == "edge":
            template = cv2.Canny(self.template, canny_low, canny_high)
            search_img = cv2.Canny(self.image, canny_low, canny_high)
        else:
            raise ValueError(f"Unknown method: {method_name}")
            
        # Apply mask if provided
        template = self._apply_mask_to_template(template, method_name)
        
        return template, search_img

    def _rotated_templates(self, template, rot_inv, multi_scale, scales):
        """Generate rotated versions of template"""
        angles = self._get_rotation_angles(rot_inv, multi_scale, scales)
        
        for angle in angles:
            if angle == 0:
                yield template
            else:
                center = (template.shape[1]//2, template.shape[0]//2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                yield cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))

    def _is_valid_template_size(self, template, search_img):
        """Check if template size is valid for matching"""
        h, w = template.shape[:2]
        img_h, img_w = search_img.shape[:2]
        
        min_size = self.params.get("min_template_size", 5)
        max_size_ratio = self.params.get("max_template_ratio", 0.8)
        
        # Template must be smaller than search image
        if h >= img_h or w >= img_w:
            return False
        
        # Skip very small templates
        if h < min_size or w < min_size:
            return False
            
        # Skip templates that are too large
        if w > img_w * max_size_ratio or h > img_h * max_size_ratio:
            return False
            
        return True

    def detect(self):
        threshold = self.params["threshold"]
        rot_inv = self.params["rotation_invariant"]
        multi_scale = self.params["multi_scale"]
        scale_r = self.params["scale_range"]
        methods = self.params["methods"]

        detected_points = []
        method = cv2.TM_CCOEFF_NORMED

        # Generate scales
        scales = self._generate_scales(multi_scale, scale_r)

        active_methods = [name for name, use in methods.items() if use]
        
        for method_name in active_methods:
            # Just-in-time preprocessing for each method
            base_template, search_img = self._prepare_method_data(method_name)

            # Multi-scale template matching
            for scale in scales:
                # Scale template if needed
                if scale != 1.0:
                    new_w = max(10, int(base_template.shape[1] * scale))
                    new_h = max(10, int(base_template.shape[0] * scale))
                    scaled_template = cv2.resize(base_template, (new_w, new_h))
                else:
                    scaled_template = base_template

                # Skip invalid template sizes
                if not self._is_valid_template_size(scaled_template, search_img):
                    continue

                # Generate rotated templates and perform matching
                for tmpl in self._rotated_templates(scaled_template, rot_inv, multi_scale, scales):
                    # Skip oversized rotated templates
                    if not self._is_valid_template_size(tmpl, search_img):
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
        nms_threshold = self.params.get("nms_threshold", 0.3)
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), nms_threshold)
        else:
            boxes = []

        detected_img = self.image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Create debug info
        debug_info = {
            "method": "Template Matching",
            "image_shape": f"{self.image.shape[1]}x{self.image.shape[0]}",
            "template_shape": f"{self.template.shape[1]}x{self.template.shape[0]}",
            "threshold": threshold,
            "multi_scale": multi_scale,
            "rotation_invariant": rot_inv,
            "mask_applied": self.mask is not None,
            "raw_detections": len(detected_points),
            "final_detections": len(boxes)
        }

        return boxes, detected_img, debug_info

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
        raw_detections = len(boxes)
        if len(boxes) > 0:
            boxes = non_max_suppression_fast(np.array(boxes), 0.4)
            
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        # Create debug info
        debug_info = {
            "method": "Color Segmentation",
            "image_shape": f"{self.image.shape[1]}x{self.image.shape[0]}",
            "template_shape": f"{self.template.shape[1]}x{self.template.shape[0]}",
            "color_tolerance": color_tolerance,
            "min_area": min_area,
            "use_watershed": use_watershed,
            "raw_detections": raw_detections,
            "final_detections": len(boxes)
        }
        
        return boxes, detected_img, debug_info
