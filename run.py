#!/usr/bin/env python
"""
Quick start script for Liveness Detection App
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Main startup function"""
    
    print("=" * 60)
    print("🎥 Liveness Detection Application - Startup Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required!")
        sys.exit(1)
    
    print("✓ Python version OK")
    
    # Check if requirements are installed
    print("\nChecking dependencies...")
    try:
        import torch
        import streamlit
        import cv2
        import mediapipe
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstalling requirements...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      check=True)
        print("✓ Dependencies installed")
    
    # Check CUDA
    print("\nChecking device...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ GPU not available, will use CPU (slower)")
    except:
        print("⚠ Could not check CUDA status")
    
    # Start Streamlit
    print("\n" + "=" * 60)
    print("Starting Streamlit app...")
    print("=" * 60)
    print("\nThe app will open at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'], check=False)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
