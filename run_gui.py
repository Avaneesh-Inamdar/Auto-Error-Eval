#!/usr/bin/env python3
"""
🚀 Grain Size Analyzer GUI Launcher
===================================

Simple launcher for the GUI version of the grain size analyzer.

Created by: Avaneesh Inamdar
Copyright (c) 2025 Avaneesh Inamdar. All rights reserved.
Author: Avaneesh Inamdar
WATERMARK: This software was developed by Avaneesh Inamdar
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter (usually comes with Python)")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n💡 Install missing packages with:")
        print("   pip install opencv-python numpy matplotlib")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🔬 Grain Size Analyzer GUI")
    print("Created by Avaneesh Inamdar © 2024")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return 1
    
    print("✅ All dependencies found!")
    print("🚀 Starting GUI application...")
    
    try:
        # Import and run the GUI
        from grain_analyzer_gui import main as run_gui
        run_gui()
        return 0
        
    except Exception as e:
        print(f"\n❌ Failed to start GUI: {e}")
        input("\nPress Enter to exit...")
        return 1

if __name__ == "__main__":
    sys.exit(main())