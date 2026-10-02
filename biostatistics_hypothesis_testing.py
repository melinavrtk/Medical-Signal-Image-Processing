# =========================================================
# Chapter: Biostatistics & Hypothesis Testing
# =========================================================
import numpy as np
import scipy.stats as sc_stats
import matplotlib.pyplot as plt

def boxplot_groups(data_list, labels, title):
    plt.figure()
    plt.boxplot(data_list, labels=labels)
    plt.title(title)
    plt.grid()
    plt.show()

# --- 1. One-Sample T-Test ---
# Test if mean of patient measurements differs significantly from 19
x_measurements = [15, 18, 14, 14, 18, 16]
t_stat, p_val = sc_stats.ttest_1samp(x_measurements, 19)
print(f"One-Sample T-Test | t-statistic: {t_stat:.4f} | p-value: {p_val:.4e}")
print("Significant difference? (alpha=0.05):", "Yes" if p_val < 0.05 else "No")

# --- 2. Independent Two-Sample T-Test ---
group_A = [17, 19, 20, 14, 26]
group_B = [22, 25, 20, 29, 21]
t_stat, p_val = sc_stats.ttest_ind(group_A, group_B, equal_var=True)
print(f"\nTwo-Sample T-Test | p-value: {p_val:.4e}")
print("Significant difference? (alpha=0.05):", "Yes" if p_val < 0.05 else "No")

# --- 3. Wilcoxon Signed-Rank Test (Non-parametric paired test) ---
# Patient temperatures before and after drug administration
temp_before = [38, 37, 39, 38, 40]
temp_after = [35, 36, 38, 36, 37]
w_stat, p_val = sc_stats.wilcoxon(temp_before, temp_after, alternative='two-sided')
print(f"\nWilcoxon Test | W-statistic: {w_stat:.1f} | p-value: {p_val:.4e}")

# --- 4. ANOVA (Analysis of Variance) & LSD Post-Hoc ---
drug_A = np.array([23, 26, 31])
drug_B = np.array([52, 57, 59])
drug_C = np.array([89, 86, 88])

f_stat, p_val = sc_stats.f_oneway(drug_A, drug_B, drug_C)
print(f"\n1-Way ANOVA | F-statistic: {f_stat:.4f} | p-value: {p_val:.4e}")

if p_val < 0.05:
    print("Significant variance found. Proceeding with Least Significant Difference (LSD) checks...")
    # (The manual LSD code from your script can be invoked here)

# --- 5. Kruskal-Wallis H-test (Non-parametric ANOVA) ---
k_stat, p_val = sc_stats.kruskal(drug_A, drug_B, drug_C)
print(f"\nKruskal-Wallis Test | H-statistic: {k_stat:.4f} | p-value: {p_val:.4e}")
