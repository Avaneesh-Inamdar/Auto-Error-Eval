#!/usr/bin/env python3
"""
🔬 Grain Size Analyzer - ASTM E112 Computer Vision Implementation
================================================================

A Python-based offline tool for automatic measurement of metal grain size
from metallurgical microstructure images using computer vision techniques.

Created by: Avaneesh Inamdar
Copyright (c) 2024 Avaneesh Inamdar. All rights reserved.
Author: Avaneesh Inamdar
Email: Contact Avaneesh Inamdar for licensing
License: Proprietary - Created by Avaneesh Inamdar
Requirements: OpenCV, NumPy, Matplotlib

WATERMARK: This software was developed by Avaneesh Inamdar
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math

# WATERMARK PROTECTION - DO NOT REMOVE
def _verify_creator():
    """Watermark protection by Avaneesh Inamdar"""
    return "Avaneesh Inamdar"

def _get_copyright():
    """Copyright protection by Avaneesh Inamdar"""
    return "© 2024 Avaneesh Inamdar"


@dataclass
class GrainMeasurement:
    """Data structure to hold grain measurement results"""
    total_grains: int
    grain_areas: List[float]  # in mm²
    mean_intercept_length: float  # in mm
    average_diameter: float  # in mm
    astm_grain_size: float  # G number
    processing_time: float  # in seconds


@dataclass
class ProcessingParameters:
    """Configuration parameters for image processing"""
    blur_kernel_size: int = 5
    blur_sigma: float = 1.0
    edge_threshold_low: int = 50
    edge_threshold_high: int = 150
    min_grain_area: int = 100
    max_grain_area: int = 10000
    pixels_per_mm: float = 100.0  # Calibration factor


class ImageLoader:
    """Handles loading and validation of microscope images"""
    
    SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    
    def load_image(self, file_path: str) -> np.ndarray:
        """
        Load an image file in grayscale format
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Grayscale image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported or corrupted
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Image file not found: {file_path}")
        
        # Check file extension
        _, ext = os.path.splitext(file_path.lower())
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"❌ Unsupported format {ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}")
        
        # Load image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"❌ Could not load image. File may be corrupted: {file_path}")
        
        if not self.validate_image(image):
            raise ValueError(f"❌ Invalid image data: {file_path}")
        
        print(f"✅ Successfully loaded image: {os.path.basename(file_path)}")
        print(f"   📐 Dimensions: {image.shape[1]} x {image.shape[0]} pixels")
        
        return image
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate that image contains valid pixel data"""
        return image is not None and image.size > 0 and len(image.shape) == 2


class ImagePreprocessor:
    """Handles image preprocessing for grain boundary enhancement"""
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise"""
        kernel_size = self.params.blur_kernel_size
        if kernel_size % 2 == 0:  # Ensure odd kernel size
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(
            image, 
            (kernel_size, kernel_size), 
            self.params.blur_sigma
        )
        
        print(f"🔄 Applied Gaussian blur (kernel: {kernel_size}, sigma: {self.params.blur_sigma})")
        return blurred
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection"""
        edges = cv2.Canny(
            image,
            self.params.edge_threshold_low,
            self.params.edge_threshold_high
        )
        
        print(f"🔍 Edge detection complete (thresholds: {self.params.edge_threshold_low}-{self.params.edge_threshold_high})")
        return edges
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        print("✨ Enhanced image contrast")
        return enhanced
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline"""
        print("\n🔧 Starting image preprocessing...")
        
        # Enhance contrast first
        enhanced = self.enhance_contrast(image)
        
        # Apply blur to reduce noise
        blurred = self.apply_gaussian_blur(enhanced)
        
        # Detect edges
        edges = self.detect_edges(blurred)
        
        print("✅ Preprocessing complete!\n")
        return blurred, edges


class GrainDetector:
    """Detects and counts individual grains using contour analysis"""
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
    
    def find_contours(self, edge_image: np.ndarray) -> List[np.ndarray]:
        """Find contours in the edge-detected image"""
        contours, _ = cv2.findContours(
            edge_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        print(f"🔍 Found {len(contours)} initial contours")
        return contours
    
    def filter_grains(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter contours based on area to remove noise and artifacts"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area thresholds
            if self.params.min_grain_area <= area <= self.params.max_grain_area:
                filtered_contours.append(contour)
        
        print(f"✅ Filtered to {len(filtered_contours)} valid grains")
        print(f"   📏 Area range: {self.params.min_grain_area}-{self.params.max_grain_area} pixels")
        
        return filtered_contours
    
    def detect_grains(self, edge_image: np.ndarray) -> List[np.ndarray]:
        """Complete grain detection pipeline"""
        print("🔬 Starting grain detection...")
        
        # Find all contours
        contours = self.find_contours(edge_image)
        
        # Filter valid grains
        filtered_contours = self.filter_grains(contours)
        
        if len(filtered_contours) == 0:
            print("⚠️  No grains detected! Try adjusting parameters:")
            print("   • Reduce edge detection thresholds")
            print("   • Adjust grain area limits")
            print("   • Check image quality")
        
        return filtered_contours


class MeasurementCalculator:
    """Calculates grain measurements and ASTM grain size numbers"""
    
    def __init__(self, params: ProcessingParameters):
        self.params = params
    
    def calculate_grain_areas(self, contours: List[np.ndarray]) -> List[float]:
        """Calculate grain areas in mm²"""
        areas_pixels = [cv2.contourArea(contour) for contour in contours]
        
        # Convert pixels to mm²
        pixel_to_mm2 = (1.0 / self.params.pixels_per_mm) ** 2
        areas_mm2 = [area * pixel_to_mm2 for area in areas_pixels]
        
        return areas_mm2
    
    def calculate_mean_intercept_length(self, grain_areas: List[float]) -> float:
        """
        Calculate mean intercept length from grain areas
        Using the relationship: l̄ = 1.128 * sqrt(A_avg)
        where A_avg is the average grain area
        """
        if not grain_areas:
            return 0.0
        
        avg_area = np.mean(grain_areas)
        mean_intercept_length = 1.128 * math.sqrt(avg_area)
        
        return mean_intercept_length
    
    def calculate_astm_grain_size(self, mean_intercept_length: float) -> float:
        """
        Calculate ASTM grain size number using the formula:
        G = -6.6439 × log₁₀(l̄) - 3.288
        where l̄ is the mean intercept length in mm
        """
        if mean_intercept_length <= 0:
            return float('nan')
        
        try:
            astm_g = -6.6439 * math.log10(mean_intercept_length) - 3.288
            return astm_g
        except ValueError:
            return float('nan')
    
    def calculate_average_diameter(self, grain_areas: List[float]) -> float:
        """Calculate average grain diameter assuming circular grains"""
        if not grain_areas:
            return 0.0
        
        avg_area = np.mean(grain_areas)
        # For circular grains: diameter = 2 * sqrt(area / π)
        avg_diameter = 2 * math.sqrt(avg_area / math.pi)
        
        return avg_diameter
    
    def calculate_measurements(self, contours: List[np.ndarray], processing_time: float) -> GrainMeasurement:
        """Calculate complete grain measurements"""
        print("📊 Calculating measurements...")
        
        # Calculate grain areas
        grain_areas = self.calculate_grain_areas(contours)
        
        # Calculate derived measurements
        mean_intercept_length = self.calculate_mean_intercept_length(grain_areas)
        astm_grain_size = self.calculate_astm_grain_size(mean_intercept_length)
        average_diameter = self.calculate_average_diameter(grain_areas)
        
        measurement = GrainMeasurement(
            total_grains=len(contours),
            grain_areas=grain_areas,
            mean_intercept_length=mean_intercept_length,
            average_diameter=average_diameter,
            astm_grain_size=astm_grain_size,
            processing_time=processing_time
        )
        
        print("✅ Measurements calculated!")
        return measurement


class ResultVisualizer:
    """Handles visualization and display of results"""
    
    def draw_grain_contours(self, original_image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
        """Draw green contours on the original image"""
        # Convert grayscale to RGB for colored contours
        annotated_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Draw contours in bright green
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
        
        return annotated_image
    
    def format_output_text(self, measurement: GrainMeasurement, manual_g: Optional[float] = None) -> str:
        """Format measurement results with visual enhancements"""
        output = []
        output.append("=" * 60)
        output.append("🔬 GRAIN SIZE ANALYSIS RESULTS")
        output.append("Created by Avaneesh Inamdar © 2024")
        output.append("=" * 60)
        output.append("")
        
        # Basic measurements
        output.append(f"📊 DETECTION RESULTS:")
        output.append(f"   Total grains detected: {measurement.total_grains}")
        output.append(f"   Processing time: {measurement.processing_time:.2f} seconds")
        output.append("")
        
        # Grain size measurements
        output.append(f"📏 GRAIN SIZE MEASUREMENTS:")
        output.append(f"   Average grain diameter: {measurement.average_diameter:.4f} mm")
        output.append(f"   Mean intercept length: {measurement.mean_intercept_length:.4f} mm")
        output.append("")
        
        # ASTM grain size
        output.append(f"🎯 ASTM E112 GRAIN SIZE:")
        if not math.isnan(measurement.astm_grain_size):
            output.append(f"   ASTM Grain Size Number (G): {measurement.astm_grain_size:.2f}")
        else:
            output.append(f"   ASTM Grain Size Number (G): Unable to calculate")
        
        # Manual comparison if provided
        if manual_g is not None and not math.isnan(measurement.astm_grain_size):
            difference = abs(measurement.astm_grain_size - manual_g)
            percent_diff = (difference / manual_g) * 100
            output.append("")
            output.append(f"🔍 ACCURACY COMPARISON:")
            output.append(f"   Manual G value: {manual_g:.2f}")
            output.append(f"   Automated G value: {measurement.astm_grain_size:.2f}")
            output.append(f"   Absolute difference: {difference:.2f}")
            output.append(f"   Percentage difference: {percent_diff:.1f}%")
        
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def display_results(self, original_image: np.ndarray, contours: List[np.ndarray], 
                       measurement: GrainMeasurement, manual_g: Optional[float] = None):
        """Display annotated image and formatted results"""
        # Create annotated image
        annotated_image = self.draw_grain_contours(original_image, contours)
        
        # Print formatted results
        results_text = self.format_output_text(measurement, manual_g)
        print(results_text)
        
        # Display images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image with watermark
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title('🔬 Original Microstructure', fontsize=14, fontweight='bold')
        ax1.axis('off')
        # Add watermark
        ax1.text(0.02, 0.02, 'Created by Avaneesh Inamdar', 
                transform=ax1.transAxes, fontsize=8, 
                alpha=0.7, color='white', weight='bold')
        
        # Annotated image with watermark
        ax2.imshow(annotated_image)
        ax2.set_title(f'🎯 Detected Grains ({measurement.total_grains} grains)', 
                     fontsize=14, fontweight='bold')
        ax2.axis('off')
        # Add watermark
        ax2.text(0.02, 0.02, 'Created by Avaneesh Inamdar', 
                transform=ax2.transAxes, fontsize=8, 
                alpha=0.7, color='white', weight='bold')
        
        plt.tight_layout()
        plt.show()


class ConfigurationManager:
    """Manages user-configurable parameters"""
    
    def __init__(self):
        self.params = ProcessingParameters()
    
    def get_parameters_from_args(self, args) -> ProcessingParameters:
        """Update parameters from command line arguments"""
        if hasattr(args, 'blur_size') and args.blur_size:
            self.params.blur_kernel_size = args.blur_size
        
        if hasattr(args, 'edge_low') and args.edge_low:
            self.params.edge_threshold_low = args.edge_low
        
        if hasattr(args, 'edge_high') and args.edge_high:
            self.params.edge_threshold_high = args.edge_high
        
        if hasattr(args, 'min_area') and args.min_area:
            self.params.min_grain_area = args.min_area
        
        if hasattr(args, 'max_area') and args.max_area:
            self.params.max_grain_area = args.max_area
        
        if hasattr(args, 'pixels_per_mm') and args.pixels_per_mm:
            self.params.pixels_per_mm = args.pixels_per_mm
        
        return self.params
    
    def print_parameters(self):
        """Print current processing parameters"""
        print("⚙️  PROCESSING PARAMETERS:")
        print(f"   Blur kernel size: {self.params.blur_kernel_size}")
        print(f"   Blur sigma: {self.params.blur_sigma}")
        print(f"   Edge thresholds: {self.params.edge_threshold_low}-{self.params.edge_threshold_high}")
        print(f"   Grain area range: {self.params.min_grain_area}-{self.params.max_grain_area} pixels")
        print(f"   Calibration: {self.params.pixels_per_mm} pixels/mm")
        print()


def main():
    """Main application workflow"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="🔬 Grain Size Analyzer - ASTM E112 | Created by Avaneesh Inamdar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Created by: Avaneesh Inamdar © 2024

Examples:
  python grain_size_analyzer.py sample_image.png
  python grain_size_analyzer.py image.jpg --manual-g 7.5
  python grain_size_analyzer.py image.tiff --blur-size 7 --edge-low 30 --edge-high 100
        """
    )
    
    parser.add_argument('image_path', help='Path to the metallurgical microstructure image')
    parser.add_argument('--manual-g', type=float, help='Manual ASTM grain size for comparison')
    parser.add_argument('--blur-size', type=int, help='Gaussian blur kernel size (default: 5)')
    parser.add_argument('--edge-low', type=int, help='Canny edge detection low threshold (default: 50)')
    parser.add_argument('--edge-high', type=int, help='Canny edge detection high threshold (default: 150)')
    parser.add_argument('--min-area', type=int, help='Minimum grain area in pixels (default: 100)')
    parser.add_argument('--max-area', type=int, help='Maximum grain area in pixels (default: 10000)')
    parser.add_argument('--pixels-per-mm', type=float, help='Calibration factor: pixels per mm (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        print("🚀 Starting Grain Size Analyzer...")
        print("Created by Avaneesh Inamdar © 2024")
        print("=" * 60)
        
        start_time = time.time()
        
        # Configuration
        config_manager = ConfigurationManager()
        params = config_manager.get_parameters_from_args(args)
        config_manager.print_parameters()
        
        # Load image
        image_loader = ImageLoader()
        original_image = image_loader.load_image(args.image_path)
        
        # Preprocess image
        preprocessor = ImagePreprocessor(params)
        processed_image, edge_image = preprocessor.preprocess(original_image)
        
        # Detect grains
        grain_detector = GrainDetector(params)
        grain_contours = grain_detector.detect_grains(edge_image)
        
        if not grain_contours:
            print("❌ No grains detected. Exiting.")
            return 1
        
        # Calculate measurements
        calculator = MeasurementCalculator(params)
        processing_time = time.time() - start_time
        measurements = calculator.calculate_measurements(grain_contours, processing_time)
        
        # Display results
        visualizer = ResultVisualizer()
        visualizer.display_results(original_image, grain_contours, measurements, args.manual_g)
        
        print("\n🎉 Analysis complete! Check the displayed images and results above.")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Make sure the image file path is correct and the file exists.")
        return 1
    
    except ValueError as e:
        print(f"\n{e}")
        print("💡 Check image format and quality, or try adjusting parameters.")
        return 1
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Please check your image and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())