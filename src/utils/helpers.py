"""
Utility functions untuk sistem chatbot
"""
import os
import sys
import json
from typing import Dict, Any

def create_directories():
    """Membuat direktori yang diperlukan jika belum ada"""
    directories = [
        "data",
        "logs",
        "src/models",
        "src/services",
        "src/api",
        "src/data_processing",
        "src/utils"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory created: {directory}")

def check_dependencies():
    """Cek apakah semua dependencies tersedia"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "sentence_transformers",
        "faiss",
        "pandas",
        "numpy"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall dengan: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True

def setup_environment():
    """Setup environment untuk development"""
    # Add src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    src_path = os.path.join(parent_dir, 'src')

    if src_path not in sys.path:
        sys.path.insert(0, parent_dir)

    return True
