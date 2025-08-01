import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from preprocessing import resize_image, apply_image_preprocessing
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
        self.use_edge = tk.BooleanVar(value=False)
        self.rotation_invariant = tk.BooleanVar(value=False)
        self.multi_scale = tk.BooleanVar(value=False)
        self.scale_range = tk.DoubleVar(value=0.2)

        # Color segmentation variables
        self.color_tolerance = tk.IntVar(value=15)  # Reduced for better precision
        self.min_area = tk.IntVar(value=200)        # Increased to filter noise
        self.erosion_iterations = tk.IntVar(value=2)  # More cleanup
        self.dilation_iterations = tk.IntVar(value=3) # Better gap filling

        # Detection method selection
        self.detection_method = tk.StringVar(value="template")
        
        # Sidebar navigation
        self.current_section = tk.StringVar(value="preprocess")
        
        # Preprocessing variables
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.gaussian_blur = tk.IntVar(value=0)
        self.sharpen = tk.BooleanVar(value=False)
        self.denoise = tk.BooleanVar(value=False)
        self.preprocessing_enabled = tk.BooleanVar(value=False)

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

        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.grid(row=0, column=1, sticky="ns", padx=(pad,0))
        right_frame.grid_propagate(False)  # Fix width

        # Create sidebar navigation frame
        sidebar_frame = ttk.Frame(right_frame, width=140)
        sidebar_frame.grid(row=0, column=0, sticky="ns", padx=(0,5))
        sidebar_frame.grid_propagate(False)

        # Create settings content frame
        content_frame = ttk.Frame(right_frame)
        content_frame.grid(row=0, column=1, sticky="nsew", padx=(5,0))
        right_frame.columnconfigure(1, weight=1)
        right_frame.rowconfigure(0, weight=1)

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
        ttk.Label(content_frame, text="Detection Settings", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0,10))

        load_btn = ttk.Button(content_frame, text="Load Image", command=self.load_image)
        load_btn.pack(fill="x", pady=(0,10))

        # Sidebar Navigation
        ttk.Label(sidebar_frame, text="Sections", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0,5))
        
        # Navigation buttons
        self.nav_buttons = {}
        
        self.nav_buttons["preprocess"] = ttk.Button(sidebar_frame, text="Image\nPreprocessing", 
                                                   command=lambda: self.switch_section("preprocess"),
                                                   width=15)
        self.nav_buttons["preprocess"].pack(fill="x", pady=2)
        
        self.nav_buttons["template"] = ttk.Button(sidebar_frame, text="Template\nMatching", 
                                                 command=lambda: self.switch_section("template"),
                                                 width=15)
        self.nav_buttons["template"].pack(fill="x", pady=2)
        
        self.nav_buttons["color"] = ttk.Button(sidebar_frame, text="Color\nSegmentation", 
                                              command=lambda: self.switch_section("color"),
                                              width=15)
        self.nav_buttons["color"].pack(fill="x", pady=2)

        # Settings content area
        self.content_area = ttk.Frame(content_frame)
        self.content_area.pack(fill="both", expand=True)

        # Create settings sections
        self.create_preprocessing_settings()
        self.create_template_settings()
        self.create_color_settings()

        # General Apply Button
        apply_frame = ttk.Frame(content_frame)
        apply_frame.pack(fill="x", pady=10)
        self.apply_btn = ttk.Button(apply_frame, text="Apply Detection", 
                                  command=self.apply_detection_general)
        self.apply_btn.pack(fill="x")

        # General buttons
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(fill="x", pady=15)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)

        self.reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_roi)
        self.reset_btn.grid(row=0, column=0, sticky="ew", padx=5)

        self.save_btn = ttk.Button(btn_frame, text="Save", command=self.save_result)
        self.save_btn.grid(row=0, column=1, sticky="ew", padx=5)

        self.update_controls_state()
        # Initialize the sidebar selection
        self.switch_section("preprocess")

    def create_preprocessing_settings(self):
        # Preprocessing Section
        self.preprocess_frame = ttk.Frame(self.content_area)

        # Enable/disable preprocessing
        enable_frame = ttk.LabelFrame(self.preprocess_frame, text="Enable Preprocessing")
        enable_frame.pack(fill="x", pady=5)
        
        self.preprocess_chk = ttk.Checkbutton(enable_frame, text="Apply image preprocessing",
                                             variable=self.preprocessing_enabled,
                                             command=self.on_preprocessing_changed)
        self.preprocess_chk.pack(anchor="w", padx=5, pady=5)

        # Basic adjustments
        basic_frame = ttk.LabelFrame(self.preprocess_frame, text="Basic Adjustments")
        basic_frame.pack(fill="x", pady=5)

        # Brightness
        ttk.Label(basic_frame, text="Brightness:").pack(anchor="w", padx=5)
        brightness_frame = ttk.Frame(basic_frame)
        brightness_frame.pack(fill="x", padx=5, pady=(0,5))
        self.brightness_slider = ttk.Scale(brightness_frame, from_=-50, to=50, orient="horizontal",
                                         variable=self.brightness, command=self.on_brightness_change)
        self.brightness_slider.pack(side="left", fill="x", expand=True)
        self.brightness_entry = ttk.Entry(brightness_frame, width=6)
        self.brightness_entry.pack(side="left", padx=(5,0))
        self.brightness_entry.insert(0, str(self.brightness.get()))
        self.brightness_entry.bind("<Return>", self.on_brightness_entry)

        # Contrast
        ttk.Label(basic_frame, text="Contrast:").pack(anchor="w", padx=5)
        contrast_frame = ttk.Frame(basic_frame)
        contrast_frame.pack(fill="x", padx=5, pady=(0,5))
        self.contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=2.0, orient="horizontal",
                                       variable=self.contrast, command=self.on_contrast_change)
        self.contrast_slider.pack(side="left", fill="x", expand=True)
        self.contrast_entry = ttk.Entry(contrast_frame, width=6)
        self.contrast_entry.pack(side="left", padx=(5,0))
        self.contrast_entry.insert(0, f"{self.contrast.get():.1f}")
        self.contrast_entry.bind("<Return>", self.on_contrast_entry)

        # Filtering
        filter_frame = ttk.LabelFrame(self.preprocess_frame, text="Noise Reduction & Enhancement")
        filter_frame.pack(fill="x", pady=5)

        # Gaussian blur
        ttk.Label(filter_frame, text="Gaussian Blur:").pack(anchor="w", padx=5)
        blur_frame = ttk.Frame(filter_frame)
        blur_frame.pack(fill="x", padx=5, pady=(0,5))
        self.blur_slider = ttk.Scale(blur_frame, from_=0, to=10, orient="horizontal",
                                   variable=self.gaussian_blur, command=self.on_blur_change)
        self.blur_slider.pack(side="left", fill="x", expand=True)
        self.blur_entry = ttk.Entry(blur_frame, width=6)
        self.blur_entry.pack(side="left", padx=(5,0))
        self.blur_entry.insert(0, str(self.gaussian_blur.get()))
        self.blur_entry.bind("<Return>", self.on_blur_entry)

        # Advanced options
        self.sharpen_chk = ttk.Checkbutton(filter_frame, text="Sharpen image",
                                          variable=self.sharpen, command=self.on_sharpen_change)
        self.sharpen_chk.pack(anchor="w", padx=5, pady=2)

        self.denoise_chk = ttk.Checkbutton(filter_frame, text="Reduce noise",
                                          variable=self.denoise, command=self.on_denoise_change)
        self.denoise_chk.pack(anchor="w", padx=5, pady=2)

        # Preview and Reset buttons
        button_frame = ttk.Frame(self.preprocess_frame)
        button_frame.pack(fill="x", pady=10)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        self.preview_btn = ttk.Button(button_frame, text="Preview Preprocessing", 
                                     command=self.preview_preprocessing)
        self.preview_btn.grid(row=0, column=0, sticky="ew", padx=(0,5))
        
        self.reset_preprocess_btn = ttk.Button(button_frame, text="Reset Settings", 
                                              command=self.reset_preprocessing_settings)
        self.reset_preprocess_btn.grid(row=0, column=1, sticky="ew", padx=(5,0))

    def create_template_settings(self):
        # Template Matching Section
        self.template_frame = ttk.Frame(self.content_area)

        app_frame = ttk.LabelFrame(self.template_frame, text="Appearance Options")
        app_frame.pack(fill="x", pady=5)

        self.grayscale_chk = ttk.Checkbutton(app_frame, text="Detect by grayscale matching",
                                            variable=self.use_grayscale, command=self.appearance_changed)
        self.grayscale_chk.pack(anchor="w", padx=5, pady=2)

        self.edge_chk = ttk.Checkbutton(app_frame, text="Detect by edge shape matching",
                                    variable=self.use_edge, command=self.appearance_changed)
        self.edge_chk.pack(anchor="w", padx=5, pady=2)

        ttk.Label(app_frame, text="(Select one or more)").pack(anchor="w", padx=5)

        flex_frame = ttk.LabelFrame(self.template_frame, text="Detection Flexibility")
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

        thresh_frame = ttk.LabelFrame(self.template_frame, text="Matching Threshold")
        thresh_frame.pack(fill="x", pady=5)

        thresh_inner_frame = ttk.Frame(thresh_frame)
        thresh_inner_frame.pack(fill="x", padx=5, pady=5)

        self.threshold_slider = ttk.Scale(thresh_inner_frame, from_=0.0, to=1.0, orient="horizontal",
                                        variable=self.threshold, command=self.on_threshold_slider)
        self.threshold_slider.pack(side="left", fill="x", expand=True)

        self.threshold_entry = ttk.Entry(thresh_inner_frame, width=6, textvariable=self.threshold_str)
        self.threshold_entry.pack(side="left", padx=(5,0))
        self.threshold_entry.bind("<Return>", self.on_threshold_entry)

    def create_color_settings(self):
        # Color Segmentation Section
        self.color_frame = ttk.Frame(self.content_area)

        color_params_frame = ttk.LabelFrame(self.color_frame, text="Color Detection Parameters")
        color_params_frame.pack(fill="x", pady=5)

        # Color tolerance
        ttk.Label(color_params_frame, text="Color Tolerance:").pack(anchor="w", padx=5)
        tolerance_frame = ttk.Frame(color_params_frame)
        tolerance_frame.pack(fill="x", padx=5, pady=(0,5))
        self.tolerance_slider = ttk.Scale(tolerance_frame, from_=5, to=50, orient="horizontal",
                                        variable=self.color_tolerance, command=self.on_tolerance_slider)
        self.tolerance_slider.pack(side="left", fill="x", expand=True)
        self.tolerance_entry = ttk.Entry(tolerance_frame, width=6)
        self.tolerance_entry.pack(side="left", padx=(5,0))
        self.tolerance_entry.insert(0, str(self.color_tolerance.get()))
        self.tolerance_entry.bind("<Return>", self.on_tolerance_entry)

        # Minimum area
        ttk.Label(color_params_frame, text="Minimum Object Area (pixels):").pack(anchor="w", padx=5)
        area_frame = ttk.Frame(color_params_frame)
        area_frame.pack(fill="x", padx=5, pady=(0,5))
        self.area_slider = ttk.Scale(area_frame, from_=50, to=2000, orient="horizontal",
                                   variable=self.min_area, command=self.on_area_slider)
        self.area_slider.pack(side="left", fill="x", expand=True)
        self.area_entry = ttk.Entry(area_frame, width=6)
        self.area_entry.pack(side="left", padx=(5,0))
        self.area_entry.insert(0, str(self.min_area.get()))
        self.area_entry.bind("<Return>", self.on_area_entry)

        # Morphological operations
        morph_frame = ttk.LabelFrame(color_params_frame, text="Noise Reduction")
        morph_frame.pack(fill="x", pady=5)

        ttk.Label(morph_frame, text="Erosion iterations:").pack(anchor="w", padx=5)
        erosion_frame = ttk.Frame(morph_frame)
        erosion_frame.pack(fill="x", padx=5, pady=(0,5))
        self.erosion_slider = ttk.Scale(erosion_frame, from_=0, to=5, orient="horizontal",
                                      variable=self.erosion_iterations, command=self.on_erosion_slider)
        self.erosion_slider.pack(side="left", fill="x", expand=True)
        self.erosion_entry = ttk.Entry(erosion_frame, width=6)
        self.erosion_entry.pack(side="left", padx=(5,0))
        self.erosion_entry.insert(0, str(self.erosion_iterations.get()))
        self.erosion_entry.bind("<Return>", self.on_erosion_entry)

        ttk.Label(morph_frame, text="Dilation iterations:").pack(anchor="w", padx=5)
        dilation_frame = ttk.Frame(morph_frame)
        dilation_frame.pack(fill="x", padx=5, pady=(0,5))
        self.dilation_slider = ttk.Scale(dilation_frame, from_=0, to=5, orient="horizontal",
                                       variable=self.dilation_iterations, command=self.on_dilation_slider)
        self.dilation_slider.pack(side="left", fill="x", expand=True)
        self.dilation_entry = ttk.Entry(dilation_frame, width=6)
        self.dilation_entry.pack(side="left", padx=(5,0))
        self.dilation_entry.insert(0, str(self.dilation_iterations.get()))
        self.dilation_entry.bind("<Return>", self.on_dilation_entry)

    def switch_section(self, section):
        """Switch to a different settings section"""
        # Hide all sections
        self.preprocess_frame.pack_forget()
        self.template_frame.pack_forget()
        self.color_frame.pack_forget()
        
        # Update button styles (highlight active)
        for name, btn in self.nav_buttons.items():
            if name == section:
                # Active button styling
                btn.config(state="disabled")  # Disabled = selected look
            else:
                # Inactive button styling
                btn.config(state="normal")
        
        # Show selected section
        if section == "preprocess":
            self.preprocess_frame.pack(fill="both", expand=True)
            self.status_label.config(text="Image Preprocessing - Enhance image quality before detection.")
        elif section == "template":
            self.template_frame.pack(fill="both", expand=True)
            self.detection_method.set("template")
            self.status_label.config(text="Template Matching - Find objects by comparing pixel patterns.")
        elif section == "color":
            self.color_frame.pack(fill="both", expand=True)
            self.detection_method.set("color")
            self.status_label.config(text="Color Segmentation - Find objects by matching colors.")
        
        self.current_section.set(section)

    def apply_detection_general(self):
        """Apply detection using the selected method"""
        method = self.detection_method.get()
        self.apply_detection(method)


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
        
        # Apply preprocessing if enabled
        if self.preprocessing_enabled.get():
            self.img = self.apply_preprocessing_to_image(self.orig_img)
        else:
            self.img = self.orig_img.copy()
            
        self.roi = None
        self.result_image = None
        self.roi_pts.clear()
        self.cropping = False
        self.draw_image(self.img)
        self.result_canvas.delete("all")
        self.count_label.config(text="Detected objects: 0")
        
        status_text = f"Loaded image: {filename.split('/')[-1] if filename else ''}"
        if self.preprocessing_enabled.get():
            status_text += " (preprocessing applied)"
        self.status_label.config(text=status_text)
        self.update_controls_state()

    def appearance_changed(self):
        if not (self.use_grayscale.get() or self.use_edge.get()):
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
        self.save_btn.config(state=state if (self.result_image is not None) else "disabled")

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

    # Color segmentation slider and entry callbacks
    def on_tolerance_slider(self, value):
        self.tolerance_entry.delete(0, tk.END)
        self.tolerance_entry.insert(0, str(int(float(value))))

    def on_tolerance_entry(self, event):
        try:
            value = int(self.tolerance_entry.get())
            value = max(5, min(50, value))
            self.color_tolerance.set(value)
            self.tolerance_entry.delete(0, tk.END)
            self.tolerance_entry.insert(0, str(value))
        except ValueError:
            self.tolerance_entry.delete(0, tk.END)
            self.tolerance_entry.insert(0, str(self.color_tolerance.get()))

    def on_area_slider(self, value):
        self.area_entry.delete(0, tk.END)
        self.area_entry.insert(0, str(int(float(value))))

    def on_area_entry(self, event):
        try:
            value = int(self.area_entry.get())
            value = max(50, min(2000, value))
            self.min_area.set(value)
            self.area_entry.delete(0, tk.END)
            self.area_entry.insert(0, str(value))
        except ValueError:
            self.area_entry.delete(0, tk.END)
            self.area_entry.insert(0, str(self.min_area.get()))

    def on_erosion_slider(self, value):
        self.erosion_entry.delete(0, tk.END)
        self.erosion_entry.insert(0, str(int(float(value))))

    def on_erosion_entry(self, event):
        try:
            value = int(self.erosion_entry.get())
            value = max(0, min(5, value))
            self.erosion_iterations.set(value)
            self.erosion_entry.delete(0, tk.END)
            self.erosion_entry.insert(0, str(value))
        except ValueError:
            self.erosion_entry.delete(0, tk.END)
            self.erosion_entry.insert(0, str(self.erosion_iterations.get()))

    def on_dilation_slider(self, value):
        self.dilation_entry.delete(0, tk.END)
        self.dilation_entry.insert(0, str(int(float(value))))

    def on_dilation_entry(self, event):
        try:
            value = int(self.dilation_entry.get())
            value = max(0, min(5, value))
            self.dilation_iterations.set(value)
            self.dilation_entry.delete(0, tk.END)
            self.dilation_entry.insert(0, str(value))
        except ValueError:
            self.dilation_entry.delete(0, tk.END)
            self.dilation_entry.insert(0, str(self.dilation_iterations.get()))

    def apply_detection(self, method="template"):
        if self.roi is None:
            messagebox.showinfo("Select Object", "Please select an object in the image first.")
            return

        # Performance warning for template matching with multiple options
        if method == "template":
            enabled_count = sum([
                self.use_grayscale.get(),
                self.use_edge.get(),
                self.rotation_invariant.get(),
                self.multi_scale.get()
            ])
            
            if enabled_count >= 3:
                result = messagebox.askyesno(
                    "Performance Warning", 
                    "You have enabled multiple template matching options which may cause slow performance.\n\n"
                    "For better performance, consider:\n"
                    "• Using only one appearance method (grayscale OR edge)\n"
                    "• Disabling rotation detection if not needed\n"
                    "• Using smaller scale range for multi-scale detection\n\n"
                    "Continue with current settings?"
                )
                if not result:
                    return

        self.detection_running = True
        self.status_label.config(text="Detecting objects... Please wait.")
        self.root.update()

        if method == "template":
            # Template matching detection
            threshold = self.threshold.get()
            rot_inv = self.rotation_invariant.get()
            multi_scale = self.multi_scale.get()
            scale_r = self.scale_range.get()

            techniques = []
            if self.use_grayscale.get():
                techniques.append("grayscale")
            if self.use_edge.get():
                techniques.append("edge")

            if not techniques:
                messagebox.showinfo("Select Technique", "Select at least one detection method.")
                self.detection_running = False
                self.status_label.config(text="")
                return

            # Use TemplateMatcher
            from detection import TemplateMatcher
            params = {
                "threshold": threshold,
                "rotation_invariant": rot_inv,
                "multi_scale": multi_scale,
                "scale_range": scale_r,
                "methods": {
                    "grayscale": "grayscale" in techniques,
                    "color": False,  # Removed color from template matching
                    "edge": "edge" in techniques
                }
            }
            matcher = TemplateMatcher(self.img, self.roi, params)
            boxes, detected_img = matcher.detect()

        elif method == "color":
            # Color segmentation detection
            from detection import ColorSegmentationMatcher
            params = {
                "color_tolerance": self.color_tolerance.get(),
                "min_area": self.min_area.get(),
                "erosion_iterations": self.erosion_iterations.get(),
                "dilation_iterations": self.dilation_iterations.get()
            }
            matcher = ColorSegmentationMatcher(self.img, self.roi, params)
            boxes, detected_img = matcher.detect()

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

    # Preprocessing callback methods
    def on_preprocessing_changed(self):
        if self.preprocessing_enabled.get():
            self.status_label.config(text="Preprocessing enabled. Adjust settings and preview changes.")
        else:
            self.status_label.config(text="Preprocessing disabled.")
            # Reset to original image if preprocessing is disabled
            if hasattr(self, 'orig_img') and self.orig_img is not None:
                self.img = self.orig_img.copy()
                self.draw_image(self.img)

    def on_brightness_change(self, value):
        self.brightness_entry.delete(0, tk.END)
        self.brightness_entry.insert(0, str(int(float(value))))
        if self.preprocessing_enabled.get():
            self.apply_preprocessing_live()

    def on_brightness_entry(self, event):
        try:
            value = int(self.brightness_entry.get())
            value = max(-50, min(50, value))
            self.brightness.set(value)
            if self.preprocessing_enabled.get():
                self.apply_preprocessing_live()
        except ValueError:
            self.brightness_entry.delete(0, tk.END)
            self.brightness_entry.insert(0, str(self.brightness.get()))

    def on_contrast_change(self, value):
        self.contrast_entry.delete(0, tk.END)
        self.contrast_entry.insert(0, f"{float(value):.1f}")
        if self.preprocessing_enabled.get():
            self.apply_preprocessing_live()

    def on_contrast_entry(self, event):
        try:
            value = float(self.contrast_entry.get())
            value = max(0.5, min(2.0, value))
            self.contrast.set(value)
            if self.preprocessing_enabled.get():
                self.apply_preprocessing_live()
        except ValueError:
            self.contrast_entry.delete(0, tk.END)
            self.contrast_entry.insert(0, f"{self.contrast.get():.1f}")

    def on_blur_change(self, value):
        self.blur_entry.delete(0, tk.END)
        self.blur_entry.insert(0, str(int(float(value))))
        if self.preprocessing_enabled.get():
            self.apply_preprocessing_live()

    def on_blur_entry(self, event):
        try:
            value = int(self.blur_entry.get())
            value = max(0, min(10, value))
            self.gaussian_blur.set(value)
            if self.preprocessing_enabled.get():
                self.apply_preprocessing_live()
        except ValueError:
            self.blur_entry.delete(0, tk.END)
            self.blur_entry.insert(0, str(self.gaussian_blur.get()))

    def on_sharpen_change(self):
        if self.preprocessing_enabled.get():
            self.apply_preprocessing_live()

    def on_denoise_change(self):
        if self.preprocessing_enabled.get():
            self.apply_preprocessing_live()

    def apply_preprocessing_live(self):
        """Apply preprocessing in real-time as user adjusts settings"""
        if not hasattr(self, 'orig_img') or self.orig_img is None:
            return
        
        processed_img = self.apply_preprocessing_to_image(self.orig_img)
        self.img = processed_img
        self.draw_image(self.img)

    def preview_preprocessing(self):
        """Manual preview button for preprocessing"""
        if not hasattr(self, 'orig_img') or self.orig_img is None:
            messagebox.showinfo("No Image", "Please load an image first.")
            return
        
        if not self.preprocessing_enabled.get():
            messagebox.showinfo("Preprocessing Disabled", "Enable preprocessing first.")
            return
        
        processed_img = self.apply_preprocessing_to_image(self.orig_img)
        self.img = processed_img
        self.draw_image(self.img)
        self.status_label.config(text="Preprocessing applied to image.")

    def apply_preprocessing_to_image(self, img):
        """Apply all preprocessing steps to an image using preprocessing module"""
        return apply_image_preprocessing(
            img,
            brightness=self.brightness.get(),
            contrast=self.contrast.get(),
            blur=self.gaussian_blur.get(),
            sharpen=self.sharpen.get(),
            denoise=self.denoise.get()
        )

    def reset_preprocessing_settings(self):
        """Reset all preprocessing settings to default values"""
        # Reset all preprocessing variables to defaults
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.gaussian_blur.set(0)
        self.sharpen.set(False)
        self.denoise.set(False)
        self.preprocessing_enabled.set(False)
        
        # Update entry fields
        self.brightness_entry.delete(0, tk.END)
        self.brightness_entry.insert(0, "0")
        
        self.contrast_entry.delete(0, tk.END)
        self.contrast_entry.insert(0, "1.0")
        
        self.blur_entry.delete(0, tk.END)
        self.blur_entry.insert(0, "0")
        
        # Reset to original image
        if hasattr(self, 'orig_img') and self.orig_img is not None:
            self.img = self.orig_img.copy()
            self.draw_image(self.img)
            # Clear any ROI if it exists
            if self.roi is not None:
                self.roi = None
                self.roi_pts.clear()
                self.cropping = False
                self.result_canvas.delete("all")
                self.count_label.config(text="Detected objects: 0")
                self.update_controls_state()
        
        self.status_label.config(text="All preprocessing settings reset to defaults.")


    

