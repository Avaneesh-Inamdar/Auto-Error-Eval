#!/usr/bin/env python3
"""
🧪 Sample Image Creator for GUI Testing
=======================================

Creates a simple sample metallurgical image for testing the GUI.

Created by: Avaneesh Inamdar
Copyright (c) 2024 Avaneesh Inamdar. All rights reserved.
Author: Avaneesh Inamdar
WATERMARK: This software was developed by Avaneesh Inamdar
"""

import cv2
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sys

def create_sample_grain_image():
    """Create a simple sample grain structure image"""
    print("🎨 Creating sample grain structure image...")
    
    # Create base image (light gray background)
    width, height = 600, 400
    image = np.ones((height, width), dtype=np.uint8) * 220
    
    # Define grain centers in a grid-like pattern with some randomness
    grain_centers = []
    spacing = 60
    
    for y in range(spacing//2, height-spacing//2, spacing):
        for x in range(spacing//2, width-spacing//2, spacing):
            # Add some randomness to positions
            offset_x = np.random.randint(-15, 15)
            offset_y = np.random.randint(-15, 15)
            grain_centers.append((x + offset_x, y + offset_y))
    
    # Draw grains
    for i, (x, y) in enumerate(grain_centers):
        # Vary grain size
        radius = np.random.randint(18, 28)
        
        # Draw grain interior with slightly different gray levels
        gray_level = 190 + (i % 4) * 8
        cv2.circle(image, (x, y), radius, gray_level, -1)
        
        # Draw grain boundary (dark)
        cv2.circle(image, (x, y), radius, 60, 2)
    
    # Add some realistic texture
    # Add noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # Slight blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # Save the image
    filename = 'sample_microstructure.png'
    success = cv2.imwrite(filename, image)
    
    if success:
        print(f"✅ Created {filename}")
        print(f"   📐 Size: {width}x{height} pixels")
        print(f"   🔬 Grains: ~{len(grain_centers)} grains")
        print(f"\n💡 Use this image to test the GUI application!")
        return filename
    else:
        print("❌ Failed to create sample image")
        return None

def main():
    """Main function"""
    print("🧪 Sample Image Creator")
    print("Created by Avaneesh Inamdar © 2024")
    print("=" * 35)
    
    try:
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Create sample image
        filename = create_sample_grain_image()
        
        if filename:
            print(f"\n🎉 Sample image ready!")
            print(f"📁 File: {filename}")
            print(f"\n🚀 Now run the GUI with: python run_gui.py")
        else:
            print("\n❌ Failed to create sample image")
            return 1
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())