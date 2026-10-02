# Medical Signal & Image Processing with Biostatistics

This repository focuses on biological signal processing (ECG), medical image manipulation, and statistical machine learning (regression) applied to clinical datasets. It utilizes Python's data science ecosystem (`numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`).

## Repository Contents

* **`biomedical_ecg_signals.py`**: ECG signal processing and analysis. Implements signal smoothing (moving average filters), SNR (Signal-to-Noise Ratio) calculations, and differential signal visualization.
* **`medical_image_processing.py`**: Medical image manipulation using `numpy` and `PIL`. Features binary thresholding, pixel value replacement, ROI (Region of Interest) inversion, and an optimized O(N*M) single-pass image histogram computation.
* **`biostatistics_hypothesis_testing.py`**: Clinical biostatistics and hypothesis testing. Implements One/Two-Sample T-Tests, Wilcoxon Signed-Rank tests, ANOVA, and Kruskal-Wallis H-tests for medical trial analysis.
* **`multiple_linear_regression_heart.py`**: Advanced multi-feature machine learning regression on heart disease data. Evaluates multiple models (Linear, Huber, Ridge, Lasso, ElasticNet), handles multicollinearity (dropping highly correlated features > 0.7), tests combinations iteratively, and plots the optimal predictive models.
