# =========================================================
# Chapter: Frequency Domain Medical Image Filtering (FFT)
# High-Pass and Band-Reject (Notch) Filters
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_1d_highpass_filter(N, ndegree, fco):
    """
    Generates a 1D Butterworth High-Pass filter symmetrically extended for FFT.
    N: Total length of the filter
    ndegree: Order of the Butterworth filter
    fco: Cut-off frequency
    """
    L = int(np.floor(N / 2)) + 1
    fh_half = np.zeros(L)
    
    for k in range(L):
        if k == 0:
            fh_half[k] = 0.0  # Block DC component
        else:
            fh_half[k] = 1 / (1 + (fco / k) ** (2 * ndegree))
            
    # Symmetric extension for Fourier Transform
    if N % 2 == 0:
        fh = np.concatenate((fh_half, fh_half[-2::-1]))
    else:
        fh = np.concatenate((fh_half, fh_half[-1::-1]))
        
    return fh

def generate_1d_bandreject_filter(N, ndegree, fco, d):
    """
    Generates a 1D Band-Reject (Notch) filter symmetrically extended for FFT.
    d: Central frequency to reject (distance from origin)
    """
    L = int(np.floor(N / 2)) + 1
    fh_half = np.zeros(L)
    
    for k in range(L):
        dist = abs(k - d)
        if dist == 0:
            fh_half[k] = 0.0  # Max attenuation at frequency 'd'
        else:
            fh_half[k] = 1 / (1 + (fco / dist) ** (2 * ndegree))
            
    if N % 2 == 0:
        fh = np.concatenate((fh_half, fh_half[-2::-1]))
    else:
        fh = np.concatenate((fh_half, fh_half[-1::-1]))
        
    return fh

def design_2d_filter(im_shape, fh_1d):
    """
    Rotates a 1D frequency profile into a 2D circular filter mask.
    """
    y, x = im_shape
    FH = np.zeros((y, x), dtype=float)
    
    # Calculate Euclidean distance from center (y/2, x/2)
    for i in range(y):
        for j in range(x):
            K = i - y / 2
            M = j - x / 2
            r = int(np.hypot(K, M) + 0.5)
            
            if r < len(fh_1d):
                FH[i, j] = fh_1d[r]
                
    # Inverse Shift to match numpy's FFT coordinate system
    return np.fft.ifftshift(FH)

def apply_fft_filter(im, FH_mask, tones=256):
    """
    Applies the 2D filter mask to the image in the Frequency Domain using FFT.
    """
    # 1. Forward 2D Fast Fourier Transform
    Fim = np.fft.fft2(im)
    
    # 2. Multiply Image Spectrum by Filter Mask (shifted to center)
    Fim_filt = Fim * np.fft.fftshift(FH_mask)
    
    # 3. Inverse 2D Fast Fourier Transform back to Spatial Domain
    im_filtered = np.real(np.fft.ifft2(Fim_filt))
    
    # 4. Normalize to valid pixel range [0, tones-1]
    im_filtered = (im_filtered - im_filtered.min()) / (im_filtered.max() - im_filtered.min()) * (tones - 1)
    
    return im_filtered.astype(np.uint8)

# =========================================================
# MAIN PROGRAM: Filter Medical Image via Frequency Domain
# =========================================================
image_path = './images/head8.bmp' # Ensure this exists in your directory

try:
    im = np.array(Image.open(image_path).convert('L'), dtype=float)
    
    M, N = im.shape
    # Maximum radial frequency distance in the image matrix
    Flength = int(np.round(np.hypot(M, N)))
    
    # Filter Parameters
    ndegree = 2
    fco = int(Flength * 0.3)  # Cutoff frequency
    d = int(Flength * 0.25)   # Center frequency for rejection
    
    # Generate 1D profiles
    fh_orig = generate_1d_highpass_filter(Flength, ndegree, fco)
    fh_br = generate_1d_bandreject_filter(Flength, ndegree, fco, d)
    
    # Create 2D Masks
    FH_orig_2D = design_2d_filter(im.shape, fh_orig)
    FH_br_2D = design_2d_filter(im.shape, fh_br)
    
    # Apply Filters via FFT
    im_highpass = apply_fft_filter(im, FH_orig_2D)
    im_bandreject = apply_fft_filter(im, FH_br_2D)
    
    # Plot Results
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Original Medical Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.plot(fh_orig, color='teal')
    plt.title('1D High-Pass Profile')
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.imshow(FH_orig_2D, cmap='gray')
    plt.title('2D High-Pass Mask')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(im_highpass, cmap='gray')
    plt.title('Filtered (High-Pass)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.plot(fh_br, color='purple')
    plt.title('1D Band-Reject Profile')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.imshow(im_bandreject, cmap='gray')
    plt.title('Filtered (Band-Reject)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Error: Image '{image_path}' not found. Please check your path.")
