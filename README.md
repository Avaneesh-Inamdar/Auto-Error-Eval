# 🔬 Grain Size Analyzer

**Created by Avaneesh Inamdar © 2024**

A comprehensive Python-based offline tool for automatic measurement of metal grain size from metallurgical microstructure images using computer vision techniques following ASTM E112 standard.

**Author:** Avaneesh Inamdar  
**Copyright:** © 2024 Avaneesh Inamdar. All rights reserved.

## ✨ Features

- **🖥️ GUI Application**: User-friendly graphical interface with point-and-click operation
- **⌨️ Command Line Tool**: Powerful CLI for batch processing and automation
- **📴 Offline Processing**: Works completely offline with no network dependencies
- **📏 ASTM E112 Compliant**: Calculates grain size numbers using the official formula
- **🔍 Computer Vision**: Uses OpenCV for automated grain boundary detection
- **🎨 Visual Results**: Shows detected grains with green contours overlay
- **⚙️ Configurable Parameters**: Adjust blur strength, edge detection thresholds, and grain size limits
- **📊 Accuracy Comparison**: Compare automated results with manual measurements
- **💾 Export Functionality**: Save results and annotated images

## 🚀 Quick Start

### Installation

1. Install Python 3.7+ and pip
2. Install required packages:
```bash
pip install -r requirements.txt
```

### GUI Application (Recommended)

```bash
# Launch the GUI application
python run_gui.py
```

Then:
1. Click "📂 Select Microstructure Image" to upload your image
2. Optionally adjust parameters in the "⚙️ Parameters" tab
3. Click "🚀 Start Analysis" to process
4. View results in the "📊 Results" tab
5. Export results and images as needed

### Command Line Usage

```bash
# Analyze a microstructure image
python grain_size_analyzer.py sample_image.png

# Compare with manual measurement
python grain_size_analyzer.py image.jpg --manual-g 7.5

# Adjust processing parameters
python grain_size_analyzer.py image.tiff --blur-size 7 --edge-low 30 --edge-high 100
```

### Create Test Images

```bash
# Generate sample images for testing
python create_sample_image.py
```

## 🎯 What You Get

### GUI Application Features:
- **Modern Interface**: Clean, tabbed layout with real-time previews
- **Interactive Parameters**: Sliders for easy adjustment of processing settings
- **Progress Tracking**: Real-time status updates during analysis
- **Side-by-side Comparison**: Original and annotated images displayed together
- **Export Options**: Save results as text files and annotated images

### Analysis Output Example:

```
============================================================
🔬 GRAIN SIZE ANALYSIS RESULTS
Created by Avaneesh Inamdar © 2024
============================================================

📊 DETECTION RESULTS:
   Total grains detected: 156
   Processing time: 2.34 seconds

📏 GRAIN SIZE MEASUREMENTS:
   Average grain diameter: 0.0234 mm
   Mean intercept length: 0.0187 mm

🎯 ASTM E112 GRAIN SIZE:
   ASTM Grain Size Number (G): 8.42

🔍 ACCURACY COMPARISON:
   Manual G value: 8.50
   Automated G value: 8.42
   Absolute difference: 0.08
   Percentage difference: 0.9%
============================================================
```

### Visual Output:
- **Original microstructure image** with clear grain boundaries
- **Annotated image** with green contours marking each detected grain
- **Embedded watermarks** crediting Avaneesh Inamdar on all generated images

## ⚙️ Parameters

### GUI Parameters (Interactive Sliders):
- **Gaussian Blur Kernel Size** (3-15): Controls noise reduction
- **Edge Detection Thresholds** (10-300): Controls grain boundary detection sensitivity
- **Grain Area Limits** (10-50000 pixels): Filters detected grains by size
- **Calibration** (10-500 pixels/mm): Converts pixel measurements to millimeters

### Command Line Parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--blur-size` | Gaussian blur kernel size | 5 | `--blur-size 7` |
| `--edge-low` | Canny edge detection low threshold | 50 | `--edge-low 30` |
| `--edge-high` | Canny edge detection high threshold | 150 | `--edge-high 100` |
| `--min-area` | Minimum grain area in pixels | 100 | `--min-area 50` |
| `--max-area` | Maximum grain area in pixels | 10000 | `--max-area 20000` |
| `--pixels-per-mm` | Calibration factor (pixels per mm) | 100 | `--pixels-per-mm 150` |
| `--manual-g` | Manual ASTM grain size for comparison | None | `--manual-g 7.5` |

## 🔧 Troubleshooting

### No grains detected?
- Reduce edge detection thresholds (`--edge-low 30 --edge-high 100`)
- Adjust grain area limits (`--min-area 50 --max-area 20000`)
- Check image quality and contrast

### Too many false detections?
- Increase edge detection thresholds
- Increase minimum grain area
- Reduce blur kernel size

### Calibration issues?
- Measure a known distance in your image
- Calculate pixels per mm: `pixels_per_mm = pixels_measured / distance_in_mm`
- Use `--pixels-per-mm` parameter

## 📋 Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib

## 🧪 ASTM E112 Formula

The tool calculates ASTM grain size number using:

```
G = -6.6439 × log₁₀(l̄) - 3.288
```

Where `l̄` is the mean intercept length in millimeters.

## 📝 License & Credits

**Created by Avaneesh Inamdar © 2025**

This software is the intellectual property of Avaneesh Inamdar. All rights reserved.

For licensing inquiries, please contact Avaneesh Inamdar.


**WATERMARK:** This application was developed by Avaneesh Inamdar
