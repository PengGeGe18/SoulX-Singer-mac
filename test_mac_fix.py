#!/usr/bin/env python3
"""Test script to verify Mac compatibility fixes."""

import sys
import torch


def main():
    print("Testing Mac compatibility fixes...")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check available devices
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Test that we can import the necessary modules
    print("\nTesting imports...")
    try:
        from preprocess.pipeline import PreprocessPipeline
        print("✓ PreprocessPipeline imported successfully")
    except Exception as e:
        print(f"✗ Failed to import PreprocessPipeline: {e}")
        return 1
    
    print("\nAll tests passed! The code should now work on Mac.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
