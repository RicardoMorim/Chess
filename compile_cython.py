#!/usr/bin/env python3
"""
Script to compile the Cython modules for the chess engine.
Run this script to build the optimized C extensions.
"""
import os
import sys
import subprocess

def main():
    print("Compiling Cython modules for chess engine...")
    
    # Run the setup.py build_ext --inplace command
    try:
        subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"])
        print("Compilation successful! Cython acceleration is now available.")
    except subprocess.CalledProcessError:
        print("Compilation failed. Check that you have Cython and a C compiler installed.")
        print("Install requirements with: pip install cython numpy")
        return False
    
    return True

if __name__ == "__main__":
    main()
