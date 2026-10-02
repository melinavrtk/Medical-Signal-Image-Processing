# =========================================================
# Chapter: Medical Image Windowing & Non-Linear Filters
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

# --- Helper Functions ---
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def rgb2gray(im):
    """ Converts an RGB image to grayscale based on standard luminance weights. """
    if len(im.shape) == 3:
        return 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
    return im

def load_image_to_gray(image_file):
    """ Reads standard image formats or DICOM files and converts to grayscale. """
    im_type = image_file[-3:].lower()
    
    if im_type == 'dcm':
        try:
            import pydicom as dicom
            dcm_data = dicom.dcmread(image_file)
            im = np.array(dcm_data.pixel_array, dtype=float)
        except ImportError:
            print("Please install pydicom (pip install pydicom) to read DICOM files.")
            return None
    else:
        im = plt.imread(image_file)
        im = np.asarray(im, dtype=float)
        
    return rgb2gray(im)

def im_plot(im, title, tones, fz=8):
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=tones-1)
    plt.title(title, fontsize=10)
    plt.axis("off")

# --- Linear Windowing Functions (Optimized with NumPy Vectorization) ---
def simple_window(im, wc, ww, image_depth, tones):
    """ Standard DICOM Windowing (Window Center / Window Width). """
    v_b = min((2.0 * wc + ww) / 2.0, image_depth)
    v_a = max(v_b - ww, 0)
    
    # Vectorized computation
    im_scaled = ((tones - 1) * (im - v_a) / (v_b - v_a))
    im_scaled = np.clip(im_scaled, 0, tones - 1)
    return np.round(im_scaled)

def broken_window(im, image_depth, tones, gray_val, im_val):
    """ Piecewise linear transformation (Broken Window). """
    condition = im <= im_val
    lower_map = (gray_val / im_val) * im
    upper_map = (((tones - 1) - (gray_val + 1)) / (image_depth - (im_val + 1))) * (im - (im_val + 1)) + (gray_val + 1)
    
    im_scaled = np.where(condition, lower_map, upper_map)
    return np.clip(np.round(im_scaled), 0, tones - 1)

def double_window(im, ww1, wl1, ww2, wl2, image_depth, tones):
    """ Double Windowing for highlighting two distinct density ranges. """
    half = (tones / 2) - 1
    
    ve1 = round((2.0 * wl1 + ww1) / 2.0)
    vs1 = max(ve1 - ww1, 0)
    
    ve2 = min(round((2.0 * wl2 + ww2) / 2.0), image_depth)
    vs2 = ve2 - ww2
    
    if vs2 < ve1:
        new_point = round((vs2 + ve1) / 2.0)
        ve1 = new_point
        vs2 = ve1

    # Vectorized logic using np.select for multiple conditions
    conditions = [
        (im < vs1),
        (im >= vs1) & (im <= ve1),
        (im > ve1) & (im < vs2),
        (im >= vs2) & (im <= ve2),
        (im > ve2)
    ]
    
    choices = [
        0,
        np.round(((half - 0) / (ve1 - vs1)) * (im - vs1) + 0.0) if ve1 != vs1 else 0,
        half + 1,
        np.round((((tones - 1) - (half + 1)) / (ve2 - vs2)) * (im - vs2) + (half + 1)) if ve2 != vs2 else half + 1,
        tones - 1
    ]
    
    im_scaled = np.select(conditions, choices, default=0)
    return im_scaled

# --- Non-Linear Transformations (Lookup Tables) ---
def form_plot_function(tones, choice):
    """ Generates Non-Linear Transformation Lookup Tables (LUT). """
    x = np.arange(tones)
    
    if choice == 0:
        w = tones - x - 1
        text = 'Inverse'
    elif choice == 1:
        r = 0.05
        w = np.log(1 + r * x)
        text = 'Logarithmic'
    elif choice == 2:
        c = 128
        w = np.exp(x) ** (1/c) - 1
        text = 'Inverse Logarithmic'
    elif choice == 3:
        gamma = 0.55
        w = x ** gamma
        text = 'Power Law (Gamma)'
    elif choice == 4:
        w = np.sin(2 * np.pi * x / (4 * (tones - 1)))
        text = 'Sine Window'
    elif choice == 5:
        w = 1 - np.exp(-x / 90)
        text = 'Exponential Window'
    elif choice == 6:
        w = 1 / (1 + np.exp(-x / 70))
        text = 'Sigmoid'
    elif choice == 7:
        w = np.cos(2 * np.pi * x / (4 * (tones - 1)))
        text = 'Cosine Window'
    elif choice == 8:
        # Avoid division by zero
        x_safe = np.where(x == 0, 1e-5, x)
        w = 1 / np.sqrt(x_safe)
        text = 'y = 1/sqrt(x)'
        
    # Normalize to [0, tones-1]
    w = (tones - 1) * ((w - np.min(w)) / (np.max(w) - np.min(w)))
    return np.round(w), text

# =========================================================
# PART A: Applying Windows to a Toy Image Matrix
# =========================================================
clear_console()
m = [
    [16, 27, 14, 20],
    [17,  9, 20, 26],
    [11, 12, 17,  8],
    [18, 25, 22, 19]
]
im_matrix = np.asarray(m, dtype=float)
print("--- INITIAL IMAGE MATRIX ---\n", im_matrix)

image_depth, tones = 32, 8

print("\n--- SIMPLE WINDOW ---")
print(simple_window(im_matrix, wc=15, ww=20, image_depth=image_depth, tones=tones))

print("\n--- BROKEN WINDOW ---")
print(broken_window(im_matrix, image_depth=image_depth, tones=tones, gray_val=4, im_val=15))

print("\n--- DOUBLE WINDOW ---")
print(double_window(im_matrix, ww1=10, wl1=10, ww2=10, wl2=25, image_depth=image_depth, tones=tones))

# =========================================================
# PART B: Non-Linear Transformations on Real Medical Image
# =========================================================
image_file = "chest.bmp" # Ensure image exists in working directory
try:
    im_real = load_image_to_gray(image_file)
    tones_real = 256
    
    print("\n[Processing Real Image Non-Linear Filters...]")
    for f_choice in range(9):
        lut, filter_name = form_plot_function(tones_real, f_choice)
        
        start_time = time.time()
        # Normalize original image to 0-255
        im_norm = np.round((tones_real - 1) * (im_real - np.min(im_real)) / (np.max(im_real) - np.min(im_real)))
        
        # O(1) Fast Lookup Table (LUT) Application instead of nested loops
        im_transformed = lut[im_norm.astype(int)]
        elapsed = time.time() - start_time
        
        print(f"Applied {filter_name} in {elapsed:.4f} seconds.")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        im_plot(im_real, 'Original Image', tones_real)
        
        plt.subplot(1, 3, 2)
        im_plot(im_transformed, f'Filter: {filter_name}', tones_real)
        
        plt.subplot(1, 3, 3)
        plt.plot(lut, color='teal')
        plt.title(f'LUT: {filter_name}', fontsize=10)
        plt.grid()
        plt.xlabel('Original Tone')
        plt.ylabel('Mapped Tone')
        plt.tight_layout()
        plt.show()

except FileNotFoundError:
    print(f"\n[Notice]: '{image_file}' not found. Place it in the directory to run visual tests.")
