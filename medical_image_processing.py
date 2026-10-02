# =========================================================
# Chapter: Medical Image Processing & Manipulation
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- 1. Image Information Extraction ---
def image_info(filename):
    im = plt.imread(filename)
    print(f"Image: {filename}")
    print(f"Width: {np.size(im, 0)} pixels | Height: {np.size(im, 1)} pixels")
    print(f"Max gray value: {np.max(im)} | Min gray value: {np.min(im)}")
    print(f"Number of pixels: {im.size} | Memory size: {im.nbytes} bytes")

# --- 2. Pixel Value Replacement ---
def pixel_replacement(filename, target_val=186, new_val=20):
    im = plt.imread(filename)
    im_proc = np.where(im == target_val, new_val, im)
    return Image.fromarray(np.asarray(im_proc, dtype="uint8"), "L")

# --- 3. Binary Thresholding (Black & White) ---
def apply_threshold(filename, threshold=100):
    im = plt.imread(filename)
    im_proc = np.where(im <= threshold, 0, 255)
    return Image.fromarray(np.asarray(im_proc, dtype="uint8"), "L")

# --- 4. ROI Inversion (Region of Interest) ---
def invert_roi(filename):
    im = plt.imread(filename)
    im_proc = np.asarray(im, dtype="float").copy()
    
    for j in range(np.size(im, 1)):
        for i in range(np.size(im, 0)):
            # Retain original colors in specific ROI bounding box
            if (120 <= j <= 250) and (90 <= i <= 160):
                continue
            else:
                im_proc[i, j] = 255 - im_proc[i, j]
                
    return Image.fromarray(np.asarray(im_proc, dtype="uint8"), "L")

# --- 5. Histogram Calculation (Optimized Single-Pass) ---
def compute_histogram(filename):
    im = plt.imread(filename)
    im_float = np.asarray(im, dtype="float")
    histogram = np.zeros(256, dtype="int")
    
    # Efficient O(N*M) single pass
    for j in range(np.size(im_float, 1)):
        for i in range(np.size(im_float, 0)):
            tone = int(im_float[i, j])
            if 0 <= tone <= 255:
                histogram[tone] += 1
                
    plt.figure(figsize=(8, 4))
    plt.bar(range(256), histogram, color='gray', width=1.0)
    plt.xlabel('Gray Tone')
    plt.ylabel('Frequency')
    plt.title('Image Histogram')
    plt.show()
