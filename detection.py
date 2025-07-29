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

        scales = [1.0]
        if multi_scale:
            step = 0.05
            scale_min = max(0.1, 1 - scale_r)
            scale_max = 1 + scale_r
            scales = np.arange(scale_min, scale_max + step, step)

        if methods["grayscale"]:
            img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if methods["edge"]:
            img_edge = cv2.Canny(self.image, 100, 200)

        def rotated_templates(template):
            for angle in range(0, 360, 15):
                M = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), angle, 1)
                yield cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))

        for method_name, use_method in methods.items():
            if not use_method:
                continue
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

            for scale in scales:
                if scale != 1.0:
                    new_w = max(1, int(base_template.shape[1] * scale))
                    new_h = max(1, int(base_template.shape[0] * scale))
                    scaled_template = cv2.resize(base_template, (new_w, new_h))
                else:
                    scaled_template = base_template

                if rot_inv:
                    templates = rotated_templates(scaled_template)
                else:
                    templates = [scaled_template]

                for tmpl in templates:
                    res = cv2.matchTemplate(search_img, tmpl, method)
                    loc = np.where(res >= threshold)
                    for pt in zip(*loc[::-1]):
                        detected_points.append((pt, tmpl.shape[::-1]))

        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        boxes = non_max_suppression_fast(np.array(boxes), 0.3)

        # Draw rectangles on a copy of the image
        detected_img = self.image.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        return boxes, detected_img
