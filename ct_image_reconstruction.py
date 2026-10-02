# =========================================================
# Chapter: CT Image Reconstruction (Radon Transform)
# Methods: Filtered Back Projection (FBP) & ART (SART)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon, iradon_sart
from PIL import Image
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def rgb2gray(rgb):
    """ Converts RGB image to grayscale. """
    if len(rgb.shape) == 3:
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    return rgb

def calculate_diagonal_rmse(img_orig, img_recon):
    """ 
    Calculates the Root Mean Squared Error (RMSE) between the diagonals 
    of the original and reconstructed images, handling dimension mismatches.
    """
    diag_orig = np.diag(img_orig)
    diag_recon = np.diag(img_recon)
    min_len = min(len(diag_orig), len(diag_recon))
    
    mse = np.mean((diag_orig[:min_len] - diag_recon[:min_len]) ** 2)
    return np.sqrt(mse)

# =========================================================
# MAIN PROGRAM: CT Reconstruction Menu
# =========================================================
clear_console()

# 1. Load and Normalize Image
image_file = "./images/head5.bmp" # Ensure the path is correct
try:
    im = np.array(Image.open(image_file), dtype=float)
    im = rgb2gray(im)
    
    # Min-Max Normalization to [0, 255]
    im = (im - np.min(im)) * (255.0 / (np.max(im) - np.min(im)))
except FileNotFoundError:
    print(f"Error: Image '{image_file}' not found. Please check your path.")
    exit()

# 2. Simulate CT Scanner Projections (Radon Transform / Sinogram)
print("[Processing] Simulating CT scan (Radon Transform)...")
N_proj = 180
theta = np.arange(0, N_proj)
sinogram = radon(im, theta=theta, circle=False)

# 3. Interactive Reconstruction Menu
print("\n--- CT IMAGE RECONSTRUCTION MENU ---")
print(" [1] Filtered Back Projection (FBP)")
print(" [2] Algebraic Reconstruction Technique (SART)")

choice = input("\nSelect reconstruction method (1 or 2): ")
reconstructed_image = None
method_title = ""

if choice == '1':
    filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
    print("\nAvailable Filters for FBP:")
    for idx, f in enumerate(filters):
        print(f" [{idx + 1}] {f}")
        
    try:
        f_choice = int(input("Select filter (1-5): "))
        selected_filter = filters[f_choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice. Defaulting to 'ramp'.")
        selected_filter = 'ramp'
        
    print(f"\n[Processing] Reconstructing using FBP ({selected_filter} filter)...")
    reconstructed_image = iradon(sinogram, theta=theta, filter_name=selected_filter, circle=False)
    method_title = f"FBP Reconstruction\nFilter: {selected_filter}"

elif choice == '2':
    try:
        iterations = int(input("\nEnter number of SART iterations (0-10): "))
    except ValueError:
        print("Invalid choice. Defaulting to 1 iteration.")
        iterations = 1
        
    print(f"\n[Processing] Reconstructing using SART ({iterations} iterations)...")
    reconstructed_image = iradon_sart(sinogram, theta=theta)
    
    for i in range(iterations - 1):
        reconstructed_image = iradon_sart(sinogram, theta=theta, image=reconstructed_image)
        
    method_title = f"SART Reconstruction\nIterations: {iterations}"

else:
    print("Invalid selection. Exiting.")
    exit()

# 4. Error Evaluation
error_rmse = calculate_diagonal_rmse(im, reconstructed_image)
print(f"\n--- RECONSTRUCTION RESULTS ---")
print(f"Method used: {method_title.replace(chr(10), ' - ')}")
print(f"Diagonal RMSE: {error_rmse:.4f}")

# Note: Based on experiments, FBP performs best with the 'ramp' filter, 
# and SART achieves minimal error with 0 or 1 iterative updates.

# 5. Visual Dashboard
plt.figure(figsize=(15, 5))

# Plot Original Image
plt.subplot(1, 3, 1)
plt.imshow(im, cmap='gray')
plt.title("Original Medical Image")
plt.axis("off")

# Plot Sinogram (CT Projections)
plt.subplot(1, 3, 2)
plt.imshow(sinogram, cmap='gray', aspect='auto', extent=(0, 180, 0, sinogram.shape[0]))
plt.title("Sinogram (Radon Transform)")
plt.xlabel("Projection Angle (degrees)")
plt.ylabel("Detector Position")

# Plot Reconstructed Image
plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image, cmap='gray')
plt.title(method_title)
plt.axis("off")

plt.tight_layout()
plt.show()
