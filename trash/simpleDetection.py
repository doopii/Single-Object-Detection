import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ObjectDetectorApp:
    def __init__(self, root, image_path, max_dim=1000):
        self.root = root
        self.root.title("Single Object Detection")

        self.orig_img = cv2.imread(image_path)
        if self.orig_img is None:
            raise FileNotFoundError(f"Image '{image_path}' not found.")
        self.orig_img = self.resize_image(self.orig_img, max_dim)
        self.img = self.orig_img.copy()

        self.roi_pts = []
        self.cropping = False
        self.roi = None

        # Params with defaults
        self.threshold = 0.8
        self.use_grayscale = tk.BooleanVar(value=True)
        self.rotation_invariant = tk.BooleanVar(value=False)
        self.multi_scale = tk.BooleanVar(value=False)
        self.scale_range = tk.DoubleVar(value=0.2)  # ±20%

        self.setup_ui()
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def resize_image(self, img, max_dim):
        h, w = img.shape[:2]
        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    def setup_ui(self):
        # Left: Image and ROI selection
        self.left_frame = ttk.Frame(self.root)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10)

        self.canvas = tk.Canvas(self.left_frame,
                                width=self.img.shape[1],
                                height=self.img.shape[0],
                                cursor="cross")
        self.canvas.pack()
        self.tk_img = None
        self.draw_image(self.img)

        # Right: Controls + results
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.grid(row=0, column=1, sticky="n", padx=10, pady=10)

        ttk.Label(self.right_frame, text="Matching Threshold (0.0 - 1.0):").pack(anchor="w")
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        ttk.Scale(self.right_frame, from_=0.0, to=1.0, orient="horizontal",
                  variable=self.threshold_var).pack(fill="x", pady=(0, 10))

        ttk.Checkbutton(self.right_frame, text="Use Grayscale Matching",
                        variable=self.use_grayscale).pack(anchor="w", pady=5)

        ttk.Checkbutton(self.right_frame, text="Enable Rotation Invariance",
                        variable=self.rotation_invariant).pack(anchor="w", pady=5)

        ttk.Checkbutton(self.right_frame, text="Enable Multi-scale Matching",
                        variable=self.multi_scale).pack(anchor="w", pady=5)

        ttk.Label(self.right_frame, text="Scale Range (±%):").pack(anchor="w")
        ttk.Scale(self.right_frame, from_=0.0, to=0.5, orient="horizontal",
                  variable=self.scale_range).pack(fill="x", pady=(0, 10))

        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(btn_frame, text="Apply Detection", command=self.apply_detection).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(btn_frame, text="Reset ROI", command=self.reset_roi).pack(side="left", expand=True, fill="x", padx=5)

        ttk.Label(self.right_frame, text="Detection Result:").pack(anchor="w")
        self.result_canvas = tk.Canvas(self.right_frame, width=self.img.shape[1], height=self.img.shape[0])
        self.result_canvas.pack()

        # Detected count label
        self.count_label = ttk.Label(self.right_frame, text="Detected objects: 0")
        self.count_label.pack(anchor="w", pady=(5, 0))

    def draw_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def draw_result(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.result_img_tk = ImageTk.PhotoImage(pil_img)
        self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_img_tk)

    # ROI mouse handlers
    def on_mouse_down(self, event):
        self.roi_pts = [(event.x, event.y)]
        self.cropping = True

    def on_mouse_drag(self, event):
        if not self.cropping:
            return
        img2 = self.img.copy()
        cv2.rectangle(img2, self.roi_pts[0], (event.x, event.y), (0, 255, 0), 2)
        self.draw_image(img2)

    def on_mouse_up(self, event):
        if not self.cropping:
            return
        self.roi_pts.append((event.x, event.y))
        self.cropping = False
        cv2.rectangle(self.img, self.roi_pts[0], self.roi_pts[1], (0, 255, 0), 2)
        self.draw_image(self.img)

        x1, y1 = self.roi_pts[0]
        x2, y2 = self.roi_pts[1]
        x1, x2 = sorted((max(0, x1), max(0, x2)))
        y1, y2 = sorted((max(0, y1), max(0, y2)))

        self.roi = self.img[y1:y2, x1:x2]

    def apply_detection(self):
        if self.roi is None:
            print("Select ROI first.")
            return

        self.threshold = self.threshold_var.get()
        use_gray = self.use_grayscale.get()
        rot_inv = self.rotation_invariant.get()
        multi_scale = self.multi_scale.get()
        scale_r = self.scale_range.get()

        if use_gray:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
            method = cv2.TM_CCOEFF_NORMED
        else:
            img_gray = None
            roi_gray = None
            method = cv2.TM_CCOEFF_NORMED

        detected_img = self.img.copy()
        detected_points = []

        def rotated_templates():
            for angle in range(0, 360, 15):
                M = cv2.getRotationMatrix2D((self.roi.shape[1]//2, self.roi.shape[0]//2), angle, 1)
                rotated = cv2.warpAffine(self.roi, M, (self.roi.shape[1], self.roi.shape[0]))
                if use_gray:
                    yield cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
                else:
                    yield rotated

        scales = [1.0]
        if multi_scale:
            step = 0.05
            scale_min = max(0.1, 1 - scale_r)
            scale_max = 1 + scale_r
            scales = np.arange(scale_min, scale_max + step, step)

        def match_template(image, template):
            return cv2.matchTemplate(image, template, method)

        for scale in scales:
            if scale != 1.0:
                new_w = max(1, int(self.roi.shape[1] * scale))
                new_h = max(1, int(self.roi.shape[0] * scale))
                scaled_roi = cv2.resize(self.roi, (new_w, new_h))
                if use_gray:
                    scaled_roi_gray = cv2.cvtColor(scaled_roi, cv2.COLOR_BGR2GRAY)
                else:
                    scaled_roi_gray = scaled_roi
            else:
                scaled_roi = self.roi
                scaled_roi_gray = roi_gray if use_gray else self.roi

            if rot_inv:
                templates = rotated_templates()
            else:
                templates = [scaled_roi_gray]

            for tmpl in templates:
                if use_gray:
                    res = match_template(img_gray, tmpl)
                else:
                    channels = cv2.split(self.img)
                    roi_channels = cv2.split(tmpl)
                    res_ch = [cv2.matchTemplate(ch, rch, method) for ch, rch in zip(channels, roi_channels)]
                    res = np.minimum(np.minimum(res_ch[0], res_ch[1]), res_ch[2])

                loc = np.where(res >= self.threshold)
                for pt in zip(*loc[::-1]):
                    detected_points.append((pt, tmpl.shape[::-1]))

        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        boxes = self.non_max_suppression_fast(np.array(boxes), 0.3)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        print(f"Detected objects: {len(boxes)}")
        self.count_label.config(text=f"Detected objects: {len(boxes)}")
        self.draw_result(detected_img)

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        boxes = boxes.astype("float")
        pick = []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            suppress = [len(idxs)-1]

            for pos in range(len(idxs)-1):
                i = idxs[pos]

                xx1 = max(x1[last], x1[i])
                yy1 = max(y1[last], y1[i])
                xx2 = min(x2[last], x2[i])
                yy2 = min(y2[last], y2[i])

                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)

                overlap = float(w * h) / area[i]

                if overlap > overlapThresh:
                    suppress.append(pos)

            idxs = np.delete(idxs, suppress)

        return boxes[pick].astype("int")

    def draw_result(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.result_img_tk = ImageTk.PhotoImage(pil_img)
        self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_img_tk)

    def reset_roi(self):
        self.img = self.orig_img.copy()
        self.roi = None
        self.roi_pts.clear()
        self.cropping = False
        self.draw_image(self.img)
        self.result_canvas.delete("all")
        self.count_label.config(text="Detected objects: 0")
        print("ROI reset. Please select new ROI.")

    # ROI mouse handlers
    def on_mouse_down(self, event):
        self.roi_pts = [(event.x, event.y)]
        self.cropping = True

    def on_mouse_drag(self, event):
        if not self.cropping:
            return
        img2 = self.img.copy()
        cv2.rectangle(img2, self.roi_pts[0], (event.x, event.y), (0, 255, 0), 2)
        self.draw_image(img2)

    def on_mouse_up(self, event):
        if not self.cropping:
            return
        self.roi_pts.append((event.x, event.y))
        self.cropping = False
        cv2.rectangle(self.img, self.roi_pts[0], self.roi_pts[1], (0, 255, 0), 2)
        self.draw_image(self.img)

        x1, y1 = self.roi_pts[0]
        x2, y2 = self.roi_pts[1]
        x1, x2 = sorted((max(0, x1), max(0, x2)))
        y1, y2 = sorted((max(0, y1), max(0, y2)))

        self.roi = self.img[y1:y2, x1:x2]

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root, "screwDrivers.jpg")  # Change filename here
    root.mainloop()
