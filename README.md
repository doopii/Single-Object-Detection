# Single Object Detection

A Python GUI application for detecting objects in images using computer vision techniques.

## Features

- **Template Matching** - Find objects by pattern comparison
- **Color Segmentation** - Detect objects by color similarity
- **Image Preprocessing** - Enhance images before detection
- **Real-time Controls** - Adjust settings with instant preview

## How to Use

1. **Load an image** → Click "Load Image"
2. **Select object** → Draw rectangle around target object
3. **Choose method** → Template Matching or Color Segmentation
4. **Adjust settings** → Use sidebar controls
5. **Detect objects** → Click "Apply Detection"

## Installation

```bash
pip install opencv-python pillow numpy
python main.py
```

## Detection Methods

### Template Matching
- Grayscale pattern matching
- Edge-based shape detection
- Rotation and scale invariant options

### Color Segmentation
- HSV and BGR color matching
- Adjustable color tolerance
- Noise reduction filters

## Screenshots

*Template Matching Interface*
![Template Matching](screenshot1.png)

*Color Segmentation Interface*
![Color Segmentation](screenshot2.png)
