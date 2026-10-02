# =========================================================
# Chapter: Advanced Frequency Domain Medical Image Filtering
# Features: Butterworth & Gaussian (Low-Pass & High-Pass), Windowing
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# --- 1. Image Enhancement & Windowing Functions ---
def normalize_image(im, tones=256):
    """ Normalizes an image matrix to [0, tones-1]. """
    im_min, im_max = np.min(im), np.max(im)
    if im_max == im_min:
        return np.zeros_like(im)
    return np.round((tones - 1) * (im - im_min) / (im_max - im_min))

def simple_window(im, wc, ww, image_depth=256, tones=256):
    """ Applies standard DICOM Window Center / Window Width contrast adjustment. """
    v_b = min((2.0 * wc + ww) / 2.0, image_depth)
    v_a = max(v_b - ww, 0)
    
    im_scaled = ((tones - 1) * (im - v_a) / (v_b - v_a))
    return np.clip(np.round(im_scaled), 0, tones - 1)

# --- 2. 1D Filter Generation ---
def generate_1d_filter(N, family='Butterworth', ftype='LP', fco=30, ndegree=2):
    """
    Generates a 1D symmetric filter profile for FFT.
    family: 'Butterworth' or 'Gaussian'
    ftype: 'LP' (Low-Pass) or 'HP' (High-Pass)
    """
    L = int(np.floor(N / 2)) + 1
    x = np.arange(L)
    
    # Base Low-Pass Profiles
    if family == 'Butterworth':
        # Avoid division by zero warnings by replacing 0 with a tiny number
        x_safe = np.where(x == 0, 1e-5, x)
        lp = 1.0 / (1.0 + (x_safe / fco)**(2 * ndegree))
        lp[0] = 1.0 # DC component
    elif family == 'Gaussian':
        lp = np.exp(-(x**2) / (2 * (fco**2)))
    else:
        raise ValueError("Unsupported filter family. Choose 'Butterworth' or 'Gaussian'.")
        
    # Invert for High-Pass
    if ftype == 'HP':
        fh_half = 1.0 - lp
    else:
        fh_half = lp  # Default to LP
        
    # Symmetric extension for Fourier Transform
    if N % 2 == 0:
        fh = np.concatenate((fh_half, fh_half[-2::-1]))
    else:
        fh = np.concatenate((fh_half, fh_half[-1::-1]))
        
    return fh, f"{family} {ftype}"

# --- 3. 2D Filter Design & FFT Application ---
def design_2d_filter(im_shape, fh_1d):
    """ Rotates the 1D frequency profile into a 2D circular filter mask. """
    y, x = im_shape
    FH = np.zeros((y, x), dtype=float)
    
    for i in range(y):
        for j in range(x):
            K = i - y / 2
            M = j - x / 2
            r = int(np.hypot(K, M) + 0.5)
            if r < len(fh_1d):
                FH[i, j] = fh_1d[r]
                
    return np.fft.ifftshift(FH)

def apply_fft_filter(im, FH_mask, tones=256):
    """ Applies the 2D filter mask using Fast Fourier Transform. """
    Fim = np.fft.fft2(im)
    Fim_filt = Fim * np.fft.fftshift(FH_mask)
    im_filtered = np.real(np.fft.ifft2(Fim_filt))
    
    return normalize_image(im_filtered, tones)

# =========================================================
# MAIN PROGRAM: Filter Pipeline Execution
# =========================================================
image_path = './images/head8.bmp'  # Change to valid image path if needed

try:
    # 1. Load Image
    im = np.array(Image.open(image_path).convert('L'), dtype=float)
    im = normalize_image(im, 256)
    
    M, N = im.shape
    Flength = int(np.round(np.hypot(M, N)))

    # 2. Filter Parameters Setup
    FILTER_TYPE = 'HP'        # 'LP' for Low-Pass, 'HP' for High-Pass
    FILTER_FAMILY = 'Butterworth' # 'Butterworth' or 'Gaussian'
    
    ndegree = 2
    fco = int(Flength * 0.1)  # Cutoff frequency
    wc, ww = 130, 180         # Windowing parameters for final display
    
    print(f"Applying {FILTER_FAMILY} {FILTER_TYPE} filter (Cutoff: {fco})...")

    # 3. Generate Filters
    fh_1d, filter_title = generate_1d_filter(Flength, family=FILTER_FAMILY, ftype=FILTER_TYPE, fco=fco, ndegree=ndegree)
    FH_2D = design_2d_filter(im.shape, fh_1d)

    # 4. Apply Filter & Post-Process
    im_filtered = apply_fft_filter(im, FH_2D, tones=256)
    im_final = simple_window(im_filtered, wc, ww, image_depth=255, tones=256)

    # 5. Visualization Dashboard
    plt.figure(figsize=(12, 10))
    
    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.title("Original Medical Image")
    plt.axis('off')

    # Final Processed Image
    plt.subplot(2, 2, 2)
    plt.imshow(im_final, cmap='gray', vmin=0, vmax=255)
    plt.title(f"Processed via Frequency Domain\nFilter: {filter_title} & Windowed")
    plt.axis('off')

    # 1D Filter Profile
    plt.subplot(2, 2, 3)
    plt.plot(fh_1d, color='teal', lw=2)
    plt.axvline(len(fh_1d)//2, color='red', linestyle='--', lw=2, label='Axis of Symmetry')
    plt.title(f"1D Profile: {filter_title}")
    plt.xlabel('Spatial Frequencies')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    # 2D Filter Mask (Spectrum)
    plt.subplot(2, 2, 4)
    plt.imshow(np.fft.fftshift(FH_2D), cmap='gray')
    plt.title(f"2D Filter Mask (FFT Spectrum)\n{filter_title}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Image '{image_path}' not found. Place a sample image in the directory.")
