# =========================================================
# Chapter: Medical Image Enhancement & Contrast Stretching
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import os

def clear_console():
    """ Clears the terminal screen. """
    os.system('cls' if os.name == 'nt' else 'clear')

def simple_display(im, image_depth, tones):
    """ Scales image intensity linearly based on the theoretical maximum depth. """
    im_scaled = np.round(((tones - 1) / (image_depth - 1)) * im)
    return im_scaled

def optimal_display(im, tones):
    """ Optimal contrast stretching (Min-Max normalization) mapped to specific tones. """
    v_mn = int(np.min(im))
    v_mx = int(np.max(im))
    im_scaled = ((tones - 1) * (im - v_mn) / (v_mx - v_mn))
    return np.around(im_scaled)

def custom_display(im, tones, v_mn, v_mx):
    """ Contrast stretching with custom bounds (Windowing) and clipping. """
    im_scaled = ((tones - 1) * (im - v_mn) / (v_mx - v_mn))
    im_scaled = np.around(im_scaled)
    # Optimized clipping: values < 0 become 0, values > (tones-1) become (tones-1)
    im_scaled = np.clip(im_scaled, 0, tones - 1)
    return im_scaled

def im_plot(im, title, tones, fz):
    """ Helper function to plot medical images accurately. """
    plt.figure(figsize=(fz, fz))
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=tones-1)
    plt.title(title)
    plt.axis("off")
    plt.show()

# =========================================================
# PART A: Theoretical Matrix Operations (Toy Example)
# =========================================================
clear_console()

matrix_data = [
    [16, 27, 14, 20],
    [17,  9, 20, 26],
    [11, 12, 17,  8],
    [18, 25, 22, 19]
]

print("--- INITIAL IMAGE MATRIX ---\n")
im_matrix = np.asarray(matrix_data, dtype=float)
print(im_matrix)

image_depth = 32  # 5-bit depth assumption
tones = 8         # Target: 8 gray tones

# 1. Simple Image Display
im_simple = simple_display(im_matrix, image_depth, tones)
print("\n--- SIMPLE IMAGE DISPLAY ---")
print(im_simple)

# 2. Optimal Image Display
im_optimal = optimal_display(im_matrix, tones)
print("\n--- OPTIMAL IMAGE DISPLAY ---")
print(im_optimal)

# 3. Custom Image Display (Custom limits: Vmin=12, Vmax=22)
im_custom = custom_display(im_matrix, tones, v_mn=12, v_mx=22)
print("\n--- CUSTOM IMAGE DISPLAY (Limits: 12-22) ---")
print(im_custom)


# =========================================================
# PART B: Real Medical Image Processing (e.g., Pelvis Scan)
# =========================================================
try:
    image_file = "Pelvis.bmp" # Ensure the image exists in the working directory
    im_real = plt.imread(image_file)
    im_real = np.asarray(im_real, dtype=float)

    image_depth_real = 256
    tones_real = 256
    figure_size = 6

    # Applying enhancement algorithms
    im_optimal_real = optimal_display(im_real, tones_real)
    im_custom_real = custom_display(im_real, tones_real, v_mn=0, v_mx=200)

    # Plotting the results
    im_plot(im_real, 'Initial Medical Image', tones_real, figure_size)
    im_plot(im_optimal_real, 'Optimal Display Method (Contrast Stretched)', tones_real, figure_size)
    im_plot(im_custom_real, 'Custom Display Method (Window: 0-200)', tones_real, figure_size)

except FileNotFoundError:
    print(f"\n[Notice]: '{image_file}' not found in the directory. Skipping visual plots.")
