# =========================================================
# Chapter: Biomedical Signal Processing (ECG Analysis)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt

def write_signal(x, filename):
    with open(filename, "w") as fid:
        for val in x:
            fid.write("%f\n" % val)

def read_signal(filename):
    with open(filename, "r") as fid:
        return np.array([float(line.strip()) for line in fid if line.strip()])

# --- 1. Basic Signal Manipulation ---
x1 = np.array([1, 0, 20, 5, -3, -8, 0, 1, 2, 1])
x2 = np.array([2, 1, 3, 3, 4, -1, 10, -4, 1, 1])

# If max(x1) > max(x2), subtract x2 from x1, else reverse
if np.max(x1) > np.max(x2):
    x3 = x1 - x2
else:
    x3 = x2 - x1

n = np.arange(0, len(x1), 1)

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(n, x1, 'b', label="Signal x1")
plt.plot(n, x2, 'r-*', label="Signal x2")
plt.grid()
plt.legend()
plt.title("Original Signals")

plt.subplot(2, 1, 2)
plt.plot(n, x3, 'k', label="Signal x3 (Difference)")
plt.grid()
plt.legend()
plt.title("Differential Signal")
plt.tight_layout()
plt.show()

# --- 2. ECG Signal Metrics (Mean, SD, SNR) ---
def signal_mean(x):
    return np.mean(x)

def signal_sd(x):
    return np.std(x, ddof=0)

def signal_snr(x):
    m = signal_mean(x)
    sd = signal_sd(x)
    return m / sd if sd != 0 else 0

# --- 3. Signal Smoothing (Moving Average) ---
def signal_smooth(x):
    x_smooth = np.empty(len(x))
    x_smooth[0] = (x[0] + x[1]) / 2
    for i in range(1, len(x) - 1):
        x_smooth[i] = (x[i - 1] + x[i] + x[i + 1]) / 3
    x_smooth[-1] = (x[-2] + x[-1]) / 2
    return x_smooth

# Example execution (assuming 'ecg.txt' exists):
# filename = 'ecg.txt'
# ecg_data = read_signal(filename)
# print(f"ECG SNR (Original): {signal_snr(ecg_data):.3f}")
# ecg_smoothed = signal_smooth(ecg_data)
# print(f"ECG SNR (Smoothed): {signal_snr(ecg_smoothed):.3f}")
