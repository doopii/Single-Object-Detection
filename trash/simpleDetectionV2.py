import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class ObjectDetectorApp:
    def __init__(self, root, image_path, max_dim=1200):
        self.root = root
        self.root.title("Single Object Detection")

        self.orig_img = cv2.imread(image_path)
        if self.orig_img is None:
            messagebox.showerror("Error", f"Image '{image_path}' not found.")
            root.destroy()
            return
        self.orig_img = self.resize_image(self.orig_img, max_dim)
        self.img = self.orig_img.copy()

        self.roi_pts = []
        self.cropping = False
        self.roi = None

        # Parameters and technique
        self.threshold = tk.DoubleVar(value=0.8)
        self.threshold_str = tk.StringVar(value="0.60")
        self.use_grayscale = tk.BooleanVar(value=True)
        self.use_color = tk.BooleanVar(value=False)
        self.use_edge = tk.BooleanVar(value=False)
        self.rotation_invariant = tk.BooleanVar(value=False)
        self.multi_scale = tk.BooleanVar(value=False)
        self.scale_range = tk.DoubleVar(value=0.2)  

        self.detection_running = False
        self.result_image = None

        self.setup_ui()
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.update_controls_state()

    def resize_image(self, img, max_dim):
        h, w = img.shape[:2]
        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h))

    def setup_ui(self):
        pad = 10

        ttk.Label(self.root, text="1. Select Object (Draw Rectangle)", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", padx=pad, pady=(pad, 0))
        ttk.Label(self.root, text="2. Detection Settings", font=("Arial", 12, "bold")).grid(row=0, column=1, sticky="w", padx=pad, pady=(pad, 0))
        ttk.Label(self.root, text="3. Detection Result", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", padx=pad, pady=(pad, 0), columnspan=2)

        self.canvas = tk.Canvas(self.root,
                                width=self.img.shape[1],
                                height=self.img.shape[0],
                                cursor="cross",
                                highlightthickness=1,
                                highlightbackground="black")
        self.canvas.grid(row=1, column=0, padx=pad, pady=pad)
        self.tk_img = None
        self.draw_image(self.img)

        ctrl_frame = ttk.Frame(self.root)
        ctrl_frame.grid(row=1, column=1, sticky="n", padx=pad, pady=pad)

        # Appearance options frame
        app_frame = ttk.LabelFrame(ctrl_frame, text="Appearance Options")
        app_frame.grid(row=0, column=0, sticky="ew", pady=5)

        self.grayscale_chk = ttk.Checkbutton(app_frame, text="Detect by grayscale matching",
                                             variable=self.use_grayscale, command=self.appearance_changed)
        self.grayscale_chk.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.color_chk = ttk.Checkbutton(app_frame, text="Detect by color matching",
                                        variable=self.use_color, command=self.appearance_changed)
        self.color_chk.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        self.edge_chk = ttk.Checkbutton(app_frame, text="Detect by edge shape matching",
                                        variable=self.use_edge, command=self.appearance_changed)
        self.edge_chk.grid(row=2, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(app_frame, text="(Select one or more)").grid(row=3, column=0, sticky="w", padx=5)

        # Detection flexibility frame
        flex_frame = ttk.LabelFrame(ctrl_frame, text="Detection Flexibility")
        flex_frame.grid(row=1, column=0, sticky="ew", pady=5)

        self.rotation_chk = ttk.Checkbutton(flex_frame, text="Detect rotated objects",
                                             variable=self.rotation_invariant)
        self.rotation_chk.grid(row=0, column=0, sticky="w", padx=5, pady=2)

        self.multiscale_chk = ttk.Checkbutton(flex_frame, text="Detect objects of different sizes",
                                               variable=self.multi_scale, command=self.multiscale_changed)
        self.multiscale_chk.grid(row=1, column=0, sticky="w", padx=5, pady=2)

        ttk.Label(flex_frame, text="Size variation range:").grid(row=2, column=0, sticky="w", padx=5)
        self.scale_slider = ttk.Scale(flex_frame, from_=0.0, to=0.5, orient="horizontal",
                                      variable=self.scale_range, state="disabled")
        self.scale_slider.grid(row=3, column=0, sticky="ew", padx=5, pady=(0,5))
        self.scale_value_label = ttk.Label(flex_frame, textvariable=tk.StringVar(value=f"{self.scale_range.get()*100:.0f}%"))
        self.scale_value_label.grid(row=3, column=1, sticky="w")
        self.scale_range.trace("w", self.update_scale_label)

        # Matching threshold frame
        thresh_frame = ttk.LabelFrame(ctrl_frame, text="Matching Threshold")
        thresh_frame.grid(row=2, column=0, sticky="ew", pady=5)

        self.threshold_slider = ttk.Scale(thresh_frame, from_=0.0, to=1.0, orient="horizontal",
                                          variable=self.threshold, command=self.on_threshold_slider)
        self.threshold_slider.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.threshold_entry = ttk.Entry(thresh_frame, width=6, textvariable=self.threshold_str)
        self.threshold_entry.grid(row=0, column=1, padx=5, pady=5)
        self.threshold_entry.bind("<Return>", self.on_threshold_entry)

        # Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.grid(row=3, column=0, pady=15, sticky="ew")
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.apply_btn = ttk.Button(btn_frame, text="Apply Detection", command=self.apply_detection)
        self.apply_btn.grid(row=0, column=0, sticky="ew", padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="Select New Object", command=self.reset_roi)
        self.reset_btn.grid(row=0, column=1, sticky="ew", padx=5)

        self.save_btn = ttk.Button(ctrl_frame, text="Save Detection Result", command=self.save_result)
        self.save_btn.grid(row=4, column=0, pady=(5,0), sticky="ew")

        # Result canvas & info
        self.result_canvas = tk.Canvas(self.root, width=self.img.shape[1], height=self.img.shape[0],
                                      highlightthickness=1, highlightbackground="black")
        self.result_canvas.grid(row=3, column=0, columnspan=2, padx=pad, pady=(0,pad))

        self.count_label = ttk.Label(self.root, text="Detected objects: 0", font=("Arial", 11, "bold"))
        self.count_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=pad)

        self.status_label = ttk.Label(self.root, text="", foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky="w", padx=pad, pady=(5,0))

        self.update_controls_state()

    def appearance_changed(self):
        # Ensure at least one technique selected; if none, re-check grayscale by default
        if not (self.use_grayscale.get() or self.use_color.get() or self.use_edge.get()):
            self.use_grayscale.set(True)
            messagebox.showinfo("Info", "At least one detection method must be selected.")
        self.status_label.config(text="Appearance options changed. Click 'Apply Detection'.")

    def multiscale_changed(self):
        if self.multi_scale.get():
            self.scale_slider.config(state="normal")
        else:
            self.scale_slider.config(state="disabled")
        self.status_label.config(text="Size variation option changed. Click 'Apply Detection'.")

    def update_scale_label(self, *args):
        percent = self.scale_range.get()*100
        self.scale_value_label.config(text=f"{percent:.0f}%")

    def update_controls_state(self):
        state = "normal" if self.roi is not None else "disabled"
        self.apply_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.save_btn.config(state=state and (self.result_image is not None))

    def draw_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def draw_result(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.result_img_tk = ImageTk.PhotoImage(pil_img)
        self.result_canvas.delete("all")
        self.result_canvas.create_image(0, 0, anchor="nw", image=self.result_img_tk)

    def on_mouse_down(self, event):
        if self.detection_running:
            return
        self.roi_pts = [(event.x, event.y)]
        self.cropping = True

    def on_mouse_drag(self, event):
        if not self.cropping or self.detection_running:
            return
        img2 = self.img.copy()
        cv2.rectangle(img2, self.roi_pts[0], (event.x, event.y), (0, 255, 0), 2)
        self.draw_image(img2)

    def on_mouse_up(self, event):
        if not self.cropping or self.detection_running:
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
        self.result_image = None
        self.count_label.config(text="Detected objects: 0")
        self.status_label.config(text="Object selected. Set parameters and click 'Apply Detection'.")
        self.update_controls_state()

    def on_threshold_slider(self, val):
        val = float(val)
        self.threshold_str.set(f"{val:.2f}")

    def on_threshold_entry(self, event):
        try:
            val = float(self.threshold_str.get())
            if 0.0 <= val <= 1.0:
                self.threshold.set(val)
            else:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Input Error", "Threshold must be a number between 0.0 and 1.0")
            self.threshold_str.set(f"{self.threshold.get():.2f}")

    def apply_detection(self):
        if self.roi is None:
            messagebox.showinfo("Select Object", "Please select an object in the image first.")
            return

        self.detection_running = True
        self.status_label.config(text="Detecting objects... Please wait.")
        self.root.update()

        threshold = self.threshold.get()
        rot_inv = self.rotation_invariant.get()
        multi_scale = self.multi_scale.get()
        scale_r = self.scale_range.get()

        # Determine which techniques to run
        techniques = []
        if self.use_grayscale.get():
            techniques.append("grayscale")
        if self.use_color.get():
            techniques.append("color")
        if self.use_edge.get():
            techniques.append("edge")

        if not techniques:
            messagebox.showinfo("Select Technique", "Select at least one detection method.")
            self.detection_running = False
            self.status_label.config(text="")
            return

        detected_img = self.img.copy()
        detected_points = []

        def rotated_templates(template):
            for angle in range(0, 360, 15):
                M = cv2.getRotationMatrix2D((template.shape[1]//2, template.shape[0]//2), angle, 1)
                rotated = cv2.warpAffine(template, M, (template.shape[1], template.shape[0]))
                yield rotated

        scales = [1.0]
        if multi_scale:
            step = 0.05
            scale_min = max(0.1, 1 - scale_r)
            scale_max = 1 + scale_r
            scales = np.arange(scale_min, scale_max + step, step)

        method = cv2.TM_CCOEFF_NORMED

        # Prepare image for grayscale if needed
        if "grayscale" in techniques:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Prepare image for edge if needed
        if "edge" in techniques:
            img_edge = cv2.Canny(self.img, 100, 200)

        for tech in techniques:
            if tech == "grayscale":
                base_template = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
                search_img = img_gray
            elif tech == "color":
                base_template = self.roi
                search_img = self.img
            elif tech == "edge":
                base_template = cv2.Canny(self.roi, 100, 200)
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

        # Non-max suppression on all detections
        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        boxes = self.non_max_suppression_fast(np.array(boxes), 0.3)

        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(detected_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        self.result_image = detected_img
        self.draw_result(detected_img)
        self.count_label.config(text=f"Detected objects: {len(boxes)}")
        self.status_label.config(text="Detection completed.")
        self.detection_running = False
        self.update_controls_state()

    def reset_roi(self):
        self.img = self.orig_img.copy()
        self.roi = None
        self.roi_pts.clear()
        self.cropping = False
        self.draw_image(self.img)
        self.result_canvas.delete("all")
        self.count_label.config(text="Detected objects: 0")
        self.status_label.config(text="ROI reset. Please select new object.")
        self.update_controls_state()

    def save_result(self):
        if self.result_image is None:
            messagebox.showinfo("No Result", "No detection result to save.")
            return

        filename = "detection_result.png"
        cv2.imwrite(filename, self.result_image)
        messagebox.showinfo("Saved", f"Detection result saved as '{filename}'")

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

    def on_threshold_slider(self, val):
        val = float(val)
        self.threshold_str.set(f"{val:.2f}")

    def on_threshold_entry(self, event):
        try:
            val = float(self.threshold_str.get())
            if 0.0 <= val <= 1.0:
                self.threshold.set(val)
            else:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Input Error", "Threshold must be a number between 0.0 and 1.0")
            self.threshold_str.set(f"{self.threshold.get():.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectorApp(root, "screwDrivers.jpg")  # Change filename here to your image file
    root.mainloop()
