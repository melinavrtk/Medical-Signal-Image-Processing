# Medical Signal & Image Processing with Biostatistics

This repository contains advanced Python implementations for biomedical signal processing (ECG), medical image enhancement, CT image reconstruction, and statistical machine learning. It leverages Python's scientific ecosystem (`numpy`, `scipy`, `skimage`, `cv2`, `pandas`, `scikit-learn`) to solve real-world biomedical engineering problems.

## 📂 Repository Contents

### 1. Medical Image Processing & Reconstruction
* **`ct_image_reconstruction.py`**: Simulates CT scanner projections (Radon Transform) and performs image reconstruction using Filtered Back Projection (FBP) and Algebraic Reconstruction Techniques (SART). Includes RMSE error evaluation.
* **`advanced_frequency_filters.py` & `frequency_domain_filtering.py`**: Frequency domain filtering using Fast Fourier Transform (FFT). Implements Butterworth, Gaussian, High-Pass, and Band-Reject (Notch) filters.
* **`medical_image_filtering.py`**: Spatial domain filtering (Convolution). Applies Laplacian, High-Emphasis, and Average masks, evaluating noise reduction to classify filters as Low-Pass (LP) or High-Pass (HP).
* **`histogram_equalization_clahe.py`**: Contrast enhancement techniques. Compares standard Histogram Equalization (HE), Cumulative Distribution Functions (CDF), and Contrast Limited Adaptive Histogram Equalization (CLAHE) using OpenCV and scikit-image.
* **`medical_windowing_filters.py`**: DICOM-style Windowing/Leveling. Implements Simple, Broken, and Double windows, along with Non-Linear Lookup Tables (LUT) like Sigmoid, Exponential, and Power Law.
* **`medical_image_processing.py`**: Fundamental operations including binary thresholding, pixel replacement, Region of Interest (ROI) inversion, and optimized histogram calculation.

### 2. Biomedical Signal Processing
* **`biomedical_ecg_signals.py`**: Processing of ECG signals. Implements moving average smoothing filters, computes Signal-to-Noise Ratio (SNR), and visualizes differential signals.

### 3. Biostatistics & Machine Learning
* **`biostatistics_hypothesis_testing.py`**: Clinical trial statistical testing. Implements One/Two-Sample T-Tests, Wilcoxon Signed-Rank tests, ANOVA, and Kruskal-Wallis H-tests.
* **`multiple_linear_regression_heart.py`**: Multi-feature regression analysis on a heart disease dataset. Evaluates multiple algorithms (Linear, Huber, Ridge, Lasso, ElasticNet), handles multicollinearity, and tests feature combinations.
