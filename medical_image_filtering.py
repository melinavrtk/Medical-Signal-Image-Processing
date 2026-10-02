# =========================================================
# Chapter: Medical Image Spatial Filtering & Noise Evaluation
# =========================================================
import numpy as np
from scipy import signal
from scipy.signal import medfilt2d

# --- 1. Define Toy Image Matrix ---
# A small 4x4 matrix representing a patch of a medical image
image = np.array([
    [16, 27, 14, 20],
    [17,  9, 20, 26],
    [11, 12, 17,  8],
    [18, 25, 22, 19]
], dtype=float)

# --- 2. Helper Functions ---
def apply_filter(img, kernel):
    """ Applies a 2D convolution filter to the image with symmetric boundaries. """
    return signal.convolve2d(img, kernel, mode='same', boundary='symm')

def compute_noise_stats(original, filtered):
    """ 
    Computes the noise variation focusing on the central 2x2 matrix.
    Returns the percentage of noise reduction to classify the filter type.
    """
    # Extract the central 2x2 region
    original_center = original[1:3, 1:3]
    filtered_center = filtered[1:3, 1:3]
    
    # Calculate standard deviation (acting as a noise proxy)
    std_orig = np.std(original_center, ddof=1)
    std_filt = np.std(filtered_center, ddof=1)
    
    # Calculate noise reduction percentage
    noise_reduction = 100 * (std_orig - std_filt) / std_orig if std_orig != 0 else 0
    return std_orig, std_filt, noise_reduction

# --- 3. Define Filter Masks (Kernels) ---
smoothing_mask = np.ones((3, 3)) / 9.0

laplacian_mask = np.array([
    [ 0,  1,  0],
    [ 1, -4,  1],
    [ 0,  1,  0]
])

high_emphasis_mask = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

# --- 4. Apply Filters ---
smoothed = apply_filter(image, smoothing_mask)

laplacian = apply_filter(image, laplacian_mask)
laplacian = np.clip(laplacian, 0, 255)  # Restrict to valid pixel range

high_emphasis = apply_filter(image, high_emphasis_mask)
high_emphasis = np.clip(high_emphasis, 0, 255)  

median_filtered = medfilt2d(image, kernel_size=3)

# Dictionary to iterate over results easily
filters = {
    "Smoothing (Average)": smoothed,
    "Laplacian": laplacian,
    "High Emphasis": high_emphasis,
    "Median": median_filtered
}

# --- 5. Evaluate and Print Results ---
print("--- ORIGINAL IMAGE MATRIX ---\n", image)

for name, result in filters.items():
    print(f"\n--- {name} Filter ---")
    print("Resulting Matrix:\n", np.round(result))
    
    # Compute and classify noise
    std_orig, std_filt, noise_reduction = compute_noise_stats(image, result)
    
    # Filter classification:
    # If noise reduction > 0, it acts as a Low-Pass (LP) smoothing filter.
    # If noise reduction <= 0, it acts as a High-Pass (HP) sharpening filter.
    filter_type = 'Low-Pass (LP)' if noise_reduction > 0 else 'High-Pass (HP)'
    
    print(f"Noise Reduction (Δ%): {noise_reduction:.2f}%")
    print(f"Filter Classification: {filter_type}")
