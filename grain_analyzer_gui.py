#!/usr/bin/env python3
"""
🔬 Grain Size Analyzer GUI - ASTM E112 Computer Vision Implementation
====================================================================

A user-friendly GUI application for automatic measurement of metal grain size
from metallurgical microstructure images using computer vision techniques.

Created by: Avaneesh Inamdar
Copyright (c) 2024 Avaneesh Inamdar. All rights reserved.
Author: Avaneesh Inamdar
Email: Contact Avaneesh Inamdar for licensing
License: Proprietary - Created by Avaneesh Inamdar
Requirements: OpenCV, NumPy, Matplotlib, tkinter

WATERMARK: This software was developed by Avaneesh Inamdar
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
import time
import threading
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

# WATERMARK PROTECTION - DO NOT REMOVE
def _verify_creator():
    """Watermark protection by Avaneesh Inamdar"""
    return "Avaneesh Inamdar"

def _get_copyright():
    """Copyright protection by Avaneesh Inamdar"""
    return "© 2024 Avaneesh Inamdar"

# Import the core classes from the original analyzer
from grain_size_analyzer import (
    GrainMeasurement, ProcessingParameters, ImageLoader, 
    ImagePreprocessor, GrainDetector, MeasurementCalculator
)


class GrainAnalyzerGUI:
    """Main GUI application for grain size analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Grain Size Analyzer - ASTM E112 | Created by Avaneesh Inamdar")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Watermark protection - Avaneesh Inamdar
        self.CREATOR = "Avaneesh Inamdar"
        self.COPYRIGHT = "© 2024 Avaneesh Inamdar"
        
        # Initialize variables
        self.current_image = None
        self.current_image_path = None
        self.analysis_results = None
        self.processing_thread = None
        
        # Create GUI components
        self.setup_gui()
        
        # Initialize processing components
        self.image_loader = ImageLoader()
        
    def setup_gui(self):
        """Set up the GUI layout"""
        # Main title with watermark
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🔬 Grain Size Analyzer - ASTM E112", 
            font=('Arial', 18, 'bold'),
            fg='white', 
            bg='#2c3e50'
        )
        title_label.pack(expand=True, pady=(5, 0))
        
        # Watermark - Avaneesh Inamdar
        watermark_label = tk.Label(
            title_frame, 
            text="Created by Avaneesh Inamdar © 2024", 
            font=('Arial', 10, 'italic'),
            fg='#ecf0f1', 
            bg='#2c3e50'
        )
        watermark_label.pack(pady=(0, 5))
        
        # Create main container with notebook for tabs
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Image Upload and Analysis
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="📸 Analysis")
        
        # Tab 2: Parameters
        self.params_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.params_tab, text="⚙️ Parameters")
        
        # Tab 3: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="📊 Results")
        
        # Setup each tab
        self.setup_analysis_tab()
        self.setup_parameters_tab()
        self.setup_results_tab()
        
    def setup_analysis_tab(self):
        """Setup the main analysis tab"""
        # Image upload section
        upload_frame = ttk.LabelFrame(self.analysis_tab, text="📁 Image Upload", padding=10)
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        # Upload button
        self.upload_btn = ttk.Button(
            upload_frame, 
            text="📂 Select Microstructure Image", 
            command=self.upload_image,
            style='Accent.TButton'
        )
        self.upload_btn.pack(pady=5)
        
        # Image info
        self.image_info_label = ttk.Label(upload_frame, text="No image selected")
        self.image_info_label.pack(pady=5)
        
        # Analysis section
        analysis_frame = ttk.LabelFrame(self.analysis_tab, text="🔬 Analysis", padding=10)
        analysis_frame.pack(fill='x', padx=10, pady=5)
        
        # Analysis controls
        controls_frame = ttk.Frame(analysis_frame)
        controls_frame.pack(fill='x', pady=5)
        
        self.analyze_btn = ttk.Button(
            controls_frame, 
            text="🚀 Start Analysis", 
            command=self.start_analysis,
            state='disabled'
        )
        self.analyze_btn.pack(side='left', padx=5)
        
        # Manual G value input
        ttk.Label(controls_frame, text="Manual G (optional):").pack(side='left', padx=(20, 5))
        self.manual_g_var = tk.StringVar()
        manual_g_entry = ttk.Entry(controls_frame, textvariable=self.manual_g_var, width=10)
        manual_g_entry.pack(side='left', padx=5)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to analyze")
        self.progress_label = ttk.Label(analysis_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=5)
        
        self.progress_bar = ttk.Progressbar(analysis_frame, mode='indeterminate')
        self.progress_bar.pack(fill='x', pady=5)
        
        # Image display area
        image_frame = ttk.LabelFrame(self.analysis_tab, text="🖼️ Image Preview", padding=10)
        image_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create matplotlib figure for image display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.patch.set_facecolor('#f0f0f0')
        
        self.ax1.set_title('Original Image')
        self.ax1.axis('off')
        self.ax2.set_title('Detected Grains')
        self.ax2.axis('off')
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, image_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def setup_parameters_tab(self):
        """Setup the parameters configuration tab"""
        # Create scrollable frame
        canvas = tk.Canvas(self.params_tab)
        scrollbar = ttk.Scrollbar(self.params_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Image Processing Parameters
        img_proc_frame = ttk.LabelFrame(scrollable_frame, text="🔧 Image Processing", padding=10)
        img_proc_frame.pack(fill='x', padx=10, pady=5)
        
        # Blur parameters
        ttk.Label(img_proc_frame, text="Gaussian Blur Kernel Size:").grid(row=0, column=0, sticky='w', pady=2)
        self.blur_size_var = tk.IntVar(value=5)
        blur_scale = ttk.Scale(img_proc_frame, from_=3, to=15, variable=self.blur_size_var, orient='horizontal')
        blur_scale.grid(row=0, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(img_proc_frame, textvariable=self.blur_size_var).grid(row=0, column=2, pady=2)
        
        # Edge detection parameters
        ttk.Label(img_proc_frame, text="Edge Detection Low Threshold:").grid(row=1, column=0, sticky='w', pady=2)
        self.edge_low_var = tk.IntVar(value=50)
        edge_low_scale = ttk.Scale(img_proc_frame, from_=10, to=200, variable=self.edge_low_var, orient='horizontal')
        edge_low_scale.grid(row=1, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(img_proc_frame, textvariable=self.edge_low_var).grid(row=1, column=2, pady=2)
        
        ttk.Label(img_proc_frame, text="Edge Detection High Threshold:").grid(row=2, column=0, sticky='w', pady=2)
        self.edge_high_var = tk.IntVar(value=150)
        edge_high_scale = ttk.Scale(img_proc_frame, from_=50, to=300, variable=self.edge_high_var, orient='horizontal')
        edge_high_scale.grid(row=2, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(img_proc_frame, textvariable=self.edge_high_var).grid(row=2, column=2, pady=2)
        
        # Grain Detection Parameters
        grain_det_frame = ttk.LabelFrame(scrollable_frame, text="🔍 Grain Detection", padding=10)
        grain_det_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(grain_det_frame, text="Minimum Grain Area (pixels):").grid(row=0, column=0, sticky='w', pady=2)
        self.min_area_var = tk.IntVar(value=100)
        min_area_scale = ttk.Scale(grain_det_frame, from_=10, to=1000, variable=self.min_area_var, orient='horizontal')
        min_area_scale.grid(row=0, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(grain_det_frame, textvariable=self.min_area_var).grid(row=0, column=2, pady=2)
        
        ttk.Label(grain_det_frame, text="Maximum Grain Area (pixels):").grid(row=1, column=0, sticky='w', pady=2)
        self.max_area_var = tk.IntVar(value=10000)
        max_area_scale = ttk.Scale(grain_det_frame, from_=1000, to=50000, variable=self.max_area_var, orient='horizontal')
        max_area_scale.grid(row=1, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(grain_det_frame, textvariable=self.max_area_var).grid(row=1, column=2, pady=2)
        
        # Calibration Parameters
        calib_frame = ttk.LabelFrame(scrollable_frame, text="📏 Calibration", padding=10)
        calib_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(calib_frame, text="Pixels per Millimeter:").grid(row=0, column=0, sticky='w', pady=2)
        self.pixels_per_mm_var = tk.DoubleVar(value=100.0)
        pixels_mm_scale = ttk.Scale(calib_frame, from_=10, to=500, variable=self.pixels_per_mm_var, orient='horizontal')
        pixels_mm_scale.grid(row=0, column=1, sticky='ew', padx=10, pady=2)
        ttk.Label(calib_frame, textvariable=self.pixels_per_mm_var).grid(row=0, column=2, pady=2)
        
        # Configure grid weights
        for frame in [img_proc_frame, grain_det_frame, calib_frame]:
            frame.columnconfigure(1, weight=1)
        
        # Reset button
        reset_btn = ttk.Button(scrollable_frame, text="🔄 Reset to Defaults", command=self.reset_parameters)
        reset_btn.pack(pady=10)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_results_tab(self):
        """Setup the results display tab"""
        # Results text area
        results_frame = ttk.LabelFrame(self.results_tab, text="📊 Analysis Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10),
            state='disabled'
        )
        self.results_text.pack(fill='both', expand=True)
        
        # Export buttons
        export_frame = ttk.Frame(results_frame)
        export_frame.pack(fill='x', pady=5)
        
        ttk.Button(export_frame, text="💾 Save Results", command=self.save_results).pack(side='left', padx=5)
        ttk.Button(export_frame, text="🖼️ Save Images", command=self.save_images).pack(side='left', padx=5)
        
    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("TIFF files", "*.tiff *.tif"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Microstructure Image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                # Load and validate image
                self.current_image = self.image_loader.load_image(file_path)
                self.current_image_path = file_path
                
                # Update UI
                filename = os.path.basename(file_path)
                self.image_info_label.config(text=f"✅ {filename} ({self.current_image.shape[1]}x{self.current_image.shape[0]})")
                self.analyze_btn.config(state='normal')
                
                # Display image
                self.display_original_image()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                
    def display_original_image(self):
        """Display the original image"""
        if self.current_image is not None:
            self.ax1.clear()
            self.ax1.imshow(self.current_image, cmap='gray')
            self.ax1.set_title('Original Microstructure')
            self.ax1.axis('off')
            # Add watermark
            self.ax1.text(0.02, 0.02, 'Created by Avaneesh Inamdar', 
                         transform=self.ax1.transAxes, fontsize=8, 
                         alpha=0.7, color='white', weight='bold')
            
            self.ax2.clear()
            self.ax2.set_title('Detected Grains (Run Analysis)')
            self.ax2.axis('off')
            
            self.canvas.draw()
            
    def get_processing_parameters(self):
        """Get current processing parameters from GUI"""
        # Ensure blur kernel size is odd
        blur_size = self.blur_size_var.get()
        if blur_size % 2 == 0:
            blur_size += 1
            
        return ProcessingParameters(
            blur_kernel_size=blur_size,
            blur_sigma=1.0,
            edge_threshold_low=self.edge_low_var.get(),
            edge_threshold_high=self.edge_high_var.get(),
            min_grain_area=self.min_area_var.get(),
            max_grain_area=self.max_area_var.get(),
            pixels_per_mm=self.pixels_per_mm_var.get()
        )
        
    def start_analysis(self):
        """Start the grain analysis in a separate thread"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Disable analysis button and start progress
        self.analyze_btn.config(state='disabled')
        self.progress_bar.start()
        self.progress_var.set("🔄 Analyzing image...")
        
        # Start analysis in separate thread to keep GUI responsive
        self.processing_thread = threading.Thread(target=self.run_analysis)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def run_analysis(self):
        """Run the actual grain analysis"""
        try:
            start_time = time.time()
            
            # Get parameters
            params = self.get_processing_parameters()
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set("🔧 Preprocessing image..."))
            
            # Preprocess image
            preprocessor = ImagePreprocessor(params)
            processed_image, edge_image = preprocessor.preprocess(self.current_image)
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set("🔍 Detecting grains..."))
            
            # Detect grains
            grain_detector = GrainDetector(params)
            grain_contours = grain_detector.detect_grains(edge_image)
            
            if not grain_contours:
                self.root.after(0, lambda: self.analysis_failed("No grains detected! Try adjusting parameters."))
                return
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set("📊 Calculating measurements..."))
            
            # Calculate measurements
            calculator = MeasurementCalculator(params)
            processing_time = time.time() - start_time
            measurements = calculator.calculate_measurements(grain_contours, processing_time)
            
            # Get manual G value if provided
            manual_g = None
            try:
                manual_g_text = self.manual_g_var.get().strip()
                if manual_g_text:
                    manual_g = float(manual_g_text)
            except ValueError:
                pass
            
            # Update UI with results
            self.root.after(0, lambda: self.analysis_completed(measurements, grain_contours, manual_g))
            
        except Exception as e:
            self.root.after(0, lambda: self.analysis_failed(f"Analysis failed: {str(e)}"))
            
    def analysis_completed(self, measurements, contours, manual_g):
        """Handle successful analysis completion"""
        self.analysis_results = {
            'measurements': measurements,
            'contours': contours,
            'manual_g': manual_g
        }
        
        # Stop progress and update status
        self.progress_bar.stop()
        self.progress_var.set(f"✅ Analysis complete! Found {measurements.total_grains} grains")
        self.analyze_btn.config(state='normal')
        
        # Display results
        self.display_results()
        self.display_annotated_image(contours)
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
        
    def analysis_failed(self, error_message):
        """Handle analysis failure"""
        self.progress_bar.stop()
        self.progress_var.set("❌ Analysis failed")
        self.analyze_btn.config(state='normal')
        
        messagebox.showerror("Analysis Failed", error_message)
        
    def display_annotated_image(self, contours):
        """Display the image with detected grain contours"""
        if self.current_image is not None:
            # Create annotated image
            annotated_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
            
            # Display images with watermark
            self.ax1.clear()
            self.ax1.imshow(self.current_image, cmap='gray')
            self.ax1.set_title('Original Microstructure')
            self.ax1.axis('off')
            # Add watermark
            self.ax1.text(0.02, 0.02, 'Created by Avaneesh Inamdar', 
                         transform=self.ax1.transAxes, fontsize=8, 
                         alpha=0.7, color='white', weight='bold')
            
            self.ax2.clear()
            self.ax2.imshow(annotated_image)
            self.ax2.set_title(f'Detected Grains ({len(contours)} grains)')
            self.ax2.axis('off')
            # Add watermark
            self.ax2.text(0.02, 0.02, 'Created by Avaneesh Inamdar', 
                         transform=self.ax2.transAxes, fontsize=8, 
                         alpha=0.7, color='white', weight='bold')
            
            self.canvas.draw()
            
    def display_results(self):
        """Display analysis results in the results tab"""
        if self.analysis_results is None:
            return
            
        measurements = self.analysis_results['measurements']
        manual_g = self.analysis_results['manual_g']
        
        # Format results text
        results_text = self.format_results_text(measurements, manual_g)
        
        # Update results display
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state='disabled')
        
    def format_results_text(self, measurement, manual_g=None):
        """Format measurement results for display"""
        lines = []
        lines.append("=" * 60)
        lines.append("🔬 GRAIN SIZE ANALYSIS RESULTS")
        lines.append("Created by Avaneesh Inamdar © 2024")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic measurements
        lines.append("📊 DETECTION RESULTS:")
        lines.append(f"   Total grains detected: {measurement.total_grains}")
        lines.append(f"   Processing time: {measurement.processing_time:.2f} seconds")
        lines.append("")
        
        # Grain size measurements
        lines.append("📏 GRAIN SIZE MEASUREMENTS:")
        lines.append(f"   Average grain diameter: {measurement.average_diameter:.4f} mm")
        lines.append(f"   Mean intercept length: {measurement.mean_intercept_length:.4f} mm")
        lines.append("")
        
        # ASTM grain size
        lines.append("🎯 ASTM E112 GRAIN SIZE:")
        if not math.isnan(measurement.astm_grain_size):
            lines.append(f"   ASTM Grain Size Number (G): {measurement.astm_grain_size:.2f}")
        else:
            lines.append("   ASTM Grain Size Number (G): Unable to calculate")
        
        # Manual comparison if provided
        if manual_g is not None and not math.isnan(measurement.astm_grain_size):
            difference = abs(measurement.astm_grain_size - manual_g)
            percent_diff = (difference / manual_g) * 100
            lines.append("")
            lines.append("🔍 ACCURACY COMPARISON:")
            lines.append(f"   Manual G value: {manual_g:.2f}")
            lines.append(f"   Automated G value: {measurement.astm_grain_size:.2f}")
            lines.append(f"   Absolute difference: {difference:.2f}")
            lines.append(f"   Percentage difference: {percent_diff:.1f}%")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
        
    def reset_parameters(self):
        """Reset all parameters to default values"""
        self.blur_size_var.set(5)
        self.edge_low_var.set(50)
        self.edge_high_var.set(150)
        self.min_area_var.set(100)
        self.max_area_var.set(10000)
        self.pixels_per_mm_var.set(100.0)
        
    def save_results(self):
        """Save analysis results to a text file"""
        if self.analysis_results is None:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                measurements = self.analysis_results['measurements']
                manual_g = self.analysis_results['manual_g']
                results_text = self.format_results_text(measurements, manual_g)
                
                with open(file_path, 'w') as f:
                    f.write(results_text)
                    
                messagebox.showinfo("Success", f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
                
    def save_images(self):
        """Save the annotated image"""
        if self.analysis_results is None or self.current_image is None:
            messagebox.showwarning("Warning", "No images to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Annotated Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                contours = self.analysis_results['contours']
                annotated_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
                cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), 2)
                
                # Convert RGB to BGR for OpenCV
                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, annotated_image_bgr)
                
                messagebox.showinfo("Success", f"Annotated image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")


def main():
    """Main function to run the GUI application"""
    # Create the main window
    root = tk.Tk()
    
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create and run the application
    app = GrainAnalyzerGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()