# Single Object Detection

A Python-based object detection application using template matching and color segmentation techniques with a professional GUI interface.

## Features

- **Template Matching**: Detect objects by pixel pattern comparison with grayscale and edge detection
- **Color Segmentation**: Detect objects by color similarity using HSV and BGR color spaces
- **Image Preprocessing**: Enhance images with brightness, contrast, blur, noise reduction, and sharpening
- **Performance Optimized**: Efficient algorithms with smart warnings for intensive operations
- **Clean Architecture**: Modular design following SDA principles with proper separation of concerns

## Detection Techniques

- **Grayscale template matching** - Pattern recognition in grayscale
- **Edge-based template matching** - Shape detection using Canny edge detection
- **Color segmentation** - Multi-color space matching (HSV + BGR)
- **Rotation invariance** - Detect rotated objects (30° increments)
- **Multi-scale detection** - Find objects of different sizes

## Project Structure

```
detection.py       - Core detection algorithms (TemplateMatcher & ColorSegmentationMatcher)
ui.py             - User interface with sidebar navigation and real-time controls
preprocessing.py  - Image processing functions (resize, brightness, contrast, etc.)
utils.py          - Utility functions (non-maximum suppression)
main.py           - Application entry point with error handling
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/doopii/single-object-detection.git
cd single-object-detection
```

2. Install dependencies:
```bash
pip install opencv-python pillow numpy tkinter
```

3. Run the application:
```bash
python main.py
```

## Usage

1. **Load Image**: Click "Load Image" to select an input image
2. **Select Object**: Draw a rectangle around the object you want to detect
3. **Choose Detection Method**: 
   - **Image Preprocessing**: Enhance image quality before detection
   - **Template Matching**: Pixel-pattern based detection with multiple options
   - **Color Segmentation**: Color-based detection with tolerance controls
4. **Adjust Parameters**: Use the sidebar to fine-tune detection settings
5. **Apply Detection**: Click "Apply Detection" to find similar objects
6. **Save Results**: Save detection results as an image file

## Academic Project

This project demonstrates professional software engineering practices:

- **Software Development & Architecture (SDA)**: 
  - Clean modular design with single responsibility principle
  - Proper separation of concerns (UI, logic, data processing)
  - Professional code organization and documentation

- **Software Quality Assurance & Testing (SQAT)**: 
  - Input validation and error handling
  - Performance optimization with user warnings
  - Robust algorithms with fallback mechanisms

<img width="650" height="758" alt="image" src="https://github.com/user-attachments/assets/dd2d38ab-c795-477c-abf4-afe804f24d2d" />

## Author

**Course**: Software Development & Architecture / Software Quality Assurance & Testing  
**Date**: August 2025
