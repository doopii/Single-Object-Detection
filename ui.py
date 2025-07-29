import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from preprocessing import resize_image
from detection import TemplateMatcher
from utils import non_max_suppression_fast

class ObjectDetectorApp:
    def __init__(self, root, max_dim=400):
        self.root = root
        self.root.title("Single Object Detection")

        self.max_dim = max_dim

        self.orig_img = None
        self.img = None
        self.result_image = None
        self.roi = None
        self.roi_pts = []
        self.cropping = False

        self.threshold = tk.DoubleVar(value=0.6)
        self.threshold_str = tk.StringVar(value="0.60")
        self.use_grayscale = tk.BooleanVar(value=True)
        self.use_color = tk.BooleanVar(value=False)
        self.use_edge = tk.BooleanVar(value=False)
        self.rotation_invariant = tk.BooleanVar(value=False)
        self.multi_scale = tk.BooleanVar(value=False)
        self.scale_range = tk.DoubleVar(value=0.2)

        self.detection_running = False

        self.setup_ui()
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.update_controls_state()

    def setup_ui(self):
        pad = 10

        # Main frames
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=pad, pady=pad)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.grid(row=0, column=1, sticky="ns", padx=(pad,0))
        right_frame.grid_propagate(False)  # Fix width

        # Left frame: canvases stacked vertically
        ttk.Label(left_frame, text="Select Object", font=("Arial", 12, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(left_frame,
                                width=600, height=250,
                                cursor="cross",
                                highlightthickness=1,
                                highlightbackground="black")
        self.canvas.pack(fill="both", expand=True, pady=(0,pad))
        self.tk_img = None

        ttk.Label(left_frame, text="Detection Result", font=("Arial", 12, "bold")).pack(anchor="w")
        self.result_canvas = tk.Canvas(left_frame,
                                    width=600, height=250,
                                    highlightthickness=1,
                                    highlightbackground="black")
        self.result_canvas.pack(fill="both", expand=True)

        self.count_label = ttk.Label(left_frame, text="Detected objects: 0", font=("Arial", 11, "bold"))
        self.count_label.pack(anchor="w", pady=(pad,0))

        self.status_label = ttk.Label(left_frame, text="", foreground="blue")
        self.status_label.pack(anchor="w", pady=(2,0))

        # Right frame: settings vertical stack
        ttk.Label(right_frame, text="Detection Settings", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0,10))

        load_btn = ttk.Button(right_frame, text="Load Image", command=self.load_image)
        load_btn.pack(fill="x", pady=(0,10))

        app_frame = ttk.LabelFrame(right_frame, text="Appearance Options")
        app_frame.pack(fill="x", pady=5)

        self.grayscale_chk = ttk.Checkbutton(app_frame, text="Detect by grayscale matching",
                                            variable=self.use_grayscale, command=self.appearance_changed)
        self.grayscale_chk.pack(anchor="w", padx=5, pady=2)

        self.color_chk = ttk.Checkbutton(app_frame, text="Detect by color matching",
                                        variable=self.use_color, command=self.appearance_changed)
        self.color_chk.pack(anchor="w", padx=5, pady=2)

        self.edge_chk = ttk.Checkbutton(app_frame, text="Detect by edge shape matching",
                                    variable=self.use_edge, command=self.appearance_changed)
        self.edge_chk.pack(anchor="w", padx=5, pady=2)

        ttk.Label(app_frame, text="(Select one or more)").pack(anchor="w", padx=5)

        flex_frame = ttk.LabelFrame(right_frame, text="Detection Flexibility")
        flex_frame.pack(fill="x", pady=5)

        self.rotation_chk = ttk.Checkbutton(flex_frame, text="Detect rotated objects",
                                            variable=self.rotation_invariant)
        self.rotation_chk.pack(anchor="w", padx=5, pady=2)

        self.multiscale_chk = ttk.Checkbutton(flex_frame, text="Detect objects of different sizes",
                                            variable=self.multi_scale, command=self.multiscale_changed)
        self.multiscale_chk.pack(anchor="w", padx=5, pady=2)

        ttk.Label(flex_frame, text="Size variation range:").pack(anchor="w", padx=5)

        scale_frame = ttk.Frame(flex_frame)
        scale_frame.pack(fill="x", padx=5, pady=(0,5))

        self.scale_slider = ttk.Scale(scale_frame, from_=0.0, to=0.5, orient="horizontal",
                                    variable=self.scale_range, state="disabled", command=self.on_scale_slider)
        self.scale_slider.pack(side="left", fill="x", expand=True)

        self.scale_entry = ttk.Entry(scale_frame, width=6)
        self.scale_entry.pack(side="left", padx=(5,0))
        self.scale_entry.insert(0, f"{self.scale_range.get():.2f}")
        self.scale_entry.bind("<Return>", self.on_scale_entry)

        thresh_frame = ttk.LabelFrame(right_frame, text="Matching Threshold")
        thresh_frame.pack(fill="x", pady=5)

        thresh_inner_frame = ttk.Frame(thresh_frame)
        thresh_inner_frame.pack(fill="x", padx=5, pady=5)

        self.threshold_slider = ttk.Scale(thresh_inner_frame, from_=0.0, to=1.0, orient="horizontal",
                                        variable=self.threshold, command=self.on_threshold_slider)
        self.threshold_slider.pack(side="left", fill="x", expand=True)

        self.threshold_entry = ttk.Entry(thresh_inner_frame, width=6, textvariable=self.threshold_str)
        self.threshold_entry.pack(side="left", padx=(5,0))
        self.threshold_entry.bind("<Return>", self.on_threshold_entry)

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill="x", pady=15)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.apply_btn = ttk.Button(btn_frame, text="Apply Detection", command=self.apply_detection)
        self.apply_btn.grid(row=0, column=0, sticky="ew", padx=5)

        self.reset_btn = ttk.Button(btn_frame, text="Select New Object", command=self.reset_roi)
        self.reset_btn.grid(row=0, column=1, sticky="ew", padx=5)

        self.save_btn = ttk.Button(right_frame, text="Save Detection Result", command=self.save_result)
        self.save_btn.pack(fill="x", pady=(5,0))

        self.update_controls_state()


    def load_image(self):
        filename = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not filename:
            return
        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", f"Failed to load image:\n{filename}")
            return
        self.load_image_from_cv_image(img, filename)

    def load_image_from_path(self, path):
        img = cv2.imread(path)
        if img is not None:
            self.load_image_from_cv_image(img, path)
        else:
            messagebox.showerror("Error", f"Failed to load image:\n{path}")

    def load_image_from_cv_image(self, img, filename=""):
        self.orig_img = resize_image(img, self.max_dim)
        self.img = self.orig_img.copy()
        self.roi = None
        self.result_image = None
        self.roi_pts.clear()
        self.cropping = False
        self.draw_image(self.img)
        self.result_canvas.delete("all")
        self.count_label.config(text="Detected objects: 0")
        self.status_label.config(text=f"Loaded image: {filename.split('/')[-1] if filename else ''}")
        self.update_controls_state()

    def appearance_changed(self):
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
        self.canvas.config(width=pil_img.width, height=pil_img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    def draw_result(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.result_img_tk = ImageTk.PhotoImage(pil_img)
        self.result_canvas.config(width=pil_img.width, height=pil_img.height)
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

    def on_scale_slider(self, val):
        val = float(val)
        self.scale_entry.delete(0, tk.END)
        self.scale_entry.insert(0, f"{val:.2f}")

    def on_scale_entry(self, event):
        try:
            val = float(self.scale_entry.get())
            if 0.0 <= val <= 0.5:
                self.scale_range.set(val)
            else:
                raise ValueError()
        except ValueError:
            messagebox.showwarning("Input Error", "Size variation must be between 0.0 and 0.5")
            self.scale_entry.delete(0, tk.END)
            self.scale_entry.insert(0, f"{self.scale_range.get():.2f}")

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

        if "grayscale" in techniques:
            img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

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

        boxes = []
        for (pt, (w, h)) in detected_points:
            boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
        boxes = non_max_suppression_fast(np.array(boxes), 0.3)

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

    def update_controls_state(self):
        state = "normal" if self.roi is not None else "disabled"
        self.apply_btn.config(state=state)
        self.reset_btn.config(state=state)
        self.save_btn.config(state=state and (self.result_image is not None))
