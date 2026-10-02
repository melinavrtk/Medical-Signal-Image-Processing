# =========================================================
# Chapter: Medical Image Histogram Equalization (HE, CDF, CLAHE)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import exposure
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# --- Helper Functions for Display ---
def im_normalize(w, tones):
    """ Normalizes array values to a specific gray-tone range [0, tones-1]. """
    w_min, w_max = np.min(w), np.max(w)
    if w_max == w_min:
        return np.zeros_like(w)
    w_scaled = (tones - 1) * (w - w_min) / (w_max - w_min)
    return np.round(w_scaled)

def im_plot(im, title, tones, fz=8):
    plt.figure(figsize=(fz, fz))
    plt.imshow(im, cmap='gray', vmin=0, vmax=tones-1)
    plt.title(title)
    plt.axis("off")
    plt.show()

def calc_histogram(im, tones):
    """ Computes the histogram of the image. """
    hist, _ = np.histogram(im.flatten(), bins=tones, range=[0, tones])
    return hist

def calc_cdf(im, tones):
    """ Computes the normalized Cumulative Distribution Function (CDF). """
    hist = calc_histogram(im, tones)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max() if cdf.max() > 0 else cdf
    return cdf_normalized

# --- Histogram Equalization Algorithms ---

def custom_histogram_equalization(im, image_depth, tones):
    """ Custom implementation of standard Histogram Equalization. """
    im_flat = im.flatten()
    hist, bins = np.histogram(im_flat, tones, [0, image_depth])
    
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0) # Mask zeros to avoid division issues
    
    # Scale CDF to [0, tones-1]
    cdf_m = (cdf_m - cdf_m.min()) * (tones - 1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Map the original image to the equalized values
    im_equalized = cdf[im.astype('uint8')]
    return im_equalized

# --- Windowing Functions (Optimized) ---

def simple_window(im, wc, ww, image_depth, tones):
    v_b = min((2.0 * wc + ww) / 2.0, image_depth)
    v_a = max(v_b - ww, 0)
    im_scaled = ((tones - 1) * (im - v_a) / (v_b - v_a))
    return np.clip(np.round(im_scaled), 0, tones - 1)

def broken_window(im, image_depth, tones, gray_val, im_val):
    condition = im <= im_val
    lower_map = (gray_val / im_val) * im
    upper_map = (((tones - 1) - (gray_val + 1)) / (image_depth - (im_val + 1))) * (im - (im_val + 1)) + (gray_val + 1)
    im_scaled = np.where(condition, lower_map, upper_map)
    return np.clip(np.round(im_scaled), 0, tones - 1)

def double_window(im, ww1, wl1, ww2, wl2, image_depth, tones):
    half = (tones / 2) - 1
    ve1 = round((2.0 * wl1 + ww1) / 2.0)
    vs1 = max(ve1 - ww1, 0)
    ve2 = min(round((2.0 * wl2 + ww2) / 2.0), image_depth)
    vs2 = ve2 - ww2
    
    if vs2 < ve1:
        ve1 = round((vs2 + ve1) / 2.0)
        vs2 = ve1

    conditions = [
        (im < vs1),
        (im >= vs1) & (im <= ve1),
        (im > ve1) & (im < vs2),
        (im >= vs2) & (im <= ve2),
        (im > ve2)
    ]
    choices = [
        0,
        np.round(((half - 0) / (ve1 - vs1)) * (im - vs1)) if ve1 != vs1 else 0,
        half + 1,
        np.round((((tones - 1) - (half + 1)) / (ve2 - vs2)) * (im - vs2) + (half + 1)) if ve2 != vs2 else half + 1,
        tones - 1
    ]
    return np.select(conditions, choices, default=0)


# =========================================================
# MAIN PROGRAM: Interactive Image & Enhancement Menu
# =========================================================
clear_console()

# 1. Image Selection Menu
image_names = ["head1.bmp", "head5.bmp", "head6.bmp", "Lung_130.bmp", "chest.bmp", "pelvis.bmp", "AA1a.bmp"]
print("Available Medical Images:")
for idx, name in enumerate(image_names):
    print(f" [{idx}] {name}")

try:
    img_choice = int(input("\nSelect an image index (0-6): "))
    if img_choice < 0 or img_choice > 6:
        raise ValueError
    image_file = image_names[img_choice]
except ValueError:
    print("Invalid choice. Defaulting to 'Lung_130.bmp'.")
    image_file = "Lung_130.bmp"

try:
    im = plt.imread(image_file)
    im = np.asarray(im, dtype=float)
except FileNotFoundError:
    print(f"\n[Error] The file {image_file} was not found in the current directory.")
    exit()

tones = 256
image_depth = 256

# --- Scikit-Image: CDF HE & CLAHE Display ---
print("\n[Processing] Generating Scikit-Image HE & CLAHE comparisons...")

plt.figure(figsize=(12, 10))

# Original
plt.subplot(3, 2, 1)
plt.imshow(im, cmap='gray', vmin=0, vmax=255)
plt.title('Initial Medical Image')
plt.axis("off")

plt.subplot(3, 2, 2)
plt.hist(im.flatten(), 256, [0, 256], color='gray', alpha=0.7)
plt.plot(calc_cdf(im, tones), 'b', linewidth=2)
plt.title('Original Histogram & CDF')
plt.grid()

# Scikit-Image Standard Histogram Equalization
im_eq_ski = exposure.equalize_hist(im)
im_eq_ski = np.round(255 * im_eq_ski / np.max(im_eq_ski))

plt.subplot(3, 2, 3)
plt.imshow(im_eq_ski, cmap='gray', vmin=0, vmax=255)
plt.title('Standard HE (Scikit-Image)')
plt.axis("off")

plt.subplot(3, 2, 4)
plt.hist(im_eq_ski.flatten(), 256, [0, 256], color='gray', alpha=0.7)
plt.plot(calc_cdf(im_eq_ski, tones), 'b', linewidth=2)
plt.title('Equalized Histogram & CDF')
plt.grid()

# Scikit-Image CLAHE (Contrast Limited Adaptive Histogram Equalization)
im_clahe_ski = exposure.equalize_adapthist(np.uint8(im), clip_limit=0.03)
im_clahe_ski = np.round(255 * im_clahe_ski / np.max(im_clahe_ski))

plt.subplot(3, 2, 5)
plt.imshow(im_clahe_ski, cmap='gray', vmin=0, vmax=255)
plt.title('CLAHE (Scikit-Image)')
plt.axis("off")

plt.subplot(3, 2, 6)
plt.hist(im_clahe_ski.flatten(), 256, [0, 256], color='gray', alpha=0.7)
plt.plot(calc_cdf(im_clahe_ski, tones), 'b', linewidth=2)
plt.title('CLAHE Histogram & CDF')
plt.grid()

plt.tight_layout()
plt.show()


# --- OpenCV: HE vs CLAHE Visualization ---
print("\n[Processing] Generating OpenCV HE vs CLAHE comparison...")
img_cv = cv.imread(image_file, cv.IMREAD_GRAYSCALE)

if img_cv is not None:
    dst_cv_eq = cv.equalizeHist(img_cv)
    
    clahe_cv = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst_cv_clahe = clahe_cv.apply(img_cv)

    # Stack images side-by-side: [Original | Standard HE | CLAHE]
    comparison_stack = np.hstack((img_cv, dst_cv_eq, dst_cv_clahe))
    
    plt.figure(figsize=(15, 6))
    plt.imshow(comparison_stack, cmap='gray')
    plt.title('OpenCV Comparison: Original | Standard HE | CLAHE')
    plt.axis('off')
    plt.show()
else:
    print("[OpenCV Error] Could not load image for OpenCV processing.")


# --- Interactive Linear Windowing Menu ---
print("\n--- LINEAR WINDOWING FILTERS ---")
print(" [1] Simple Window")
print(" [2] Broken Window")
print(" [3] Double Window")
print(" [4] Skip Windowing")

try:
    w_choice = int(input("Select a windowing method (1-4): "))
except ValueError:
    w_choice = 4

if w_choice == 1:
    wc, ww = 50, 250
    im_proc = simple_window(im, wc, ww, image_depth, tones)
    im_plot(im_proc, 'Simple Window Method', tones, fz=8)
    
elif w_choice == 2:
    gray_val, im_val = 128, 70
    im_proc = broken_window(im, image_depth, tones, gray_val, im_val)
    im_plot(im_proc, 'Broken Window Method', tones, fz=8)
    
elif w_choice == 3:
    ww1, wl1 = 100, 50
    ww2, wl2 = 100, 150
    im_proc = double_window(im, ww1, wl1, ww2, wl2, image_depth, tones)
    im_plot(im_proc, 'Double Window Method', tones, fz=8)
else:
    print("Exiting image processing menu.")
