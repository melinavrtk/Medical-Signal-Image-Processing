# =========================================================
# Chapter: Linear & Multi-Feature Regression Analysis
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from itertools import combinations

# --- 1. Simple Linear Regression Example (Kidney Length vs Age) ---
# Equation: y = a * x + b
age_samples = np.array([30, 40, 50, 60], dtype=float)
length_samples = np.array([105, 100, 95, 80], dtype=float)
target_age = 55
target_length = 115

# Manual OLS Computation
x_mean, y_mean = np.mean(age_samples), np.mean(length_samples)
slope_a = np.sum((age_samples - x_mean) * (length_samples - y_mean)) / np.sum((age_samples - x_mean)**2)
intercept_b = y_mean - slope_a * x_mean

predicted_length = intercept_b + slope_a * target_age
SEM = 3.19
CI_low = predicted_length - 2 * SEM
CI_up = predicted_length + 2 * SEM

print(f"Kidney Regression Model: y = {slope_a:.4f}x + {intercept_b:.4f}")
print(f"Predicted length for age {target_age}: {predicted_length:.2f}")
print(f"Confidence Interval: [{CI_low:.2f} - {CI_up:.2f}]")

if target_length < CI_low or target_length > CI_up:
    print(f"Measurement {target_length} is OUTSIDE the normal range.")
else:
    print(f"Measurement {target_length} is WITHIN the normal range.")

# --- 2. Advanced Multi-Regressor Analysis (Heart Disease Dataset) ---
def get_regressor(choice_id):
    regressors = {
        1: (LinearRegression(), "Linear Regression"),
        2: (HuberRegressor(), "Huber Regressor"),
        3: (Ridge(), "Ridge Regressor"),
        4: (Lasso(), "Lasso Regressor"),
        5: (ElasticNet(), "ElasticNet")
    }
    return regressors.get(choice_id, (LinearRegression(), "Linear Regression"))

def get_feature_combinations(n_features, max_comb_size):
    features = list(range(n_features))
    return [list(c) for i in range(1, max_comb_size + 1) for c in combinations(features, i)]

# Ensure 'heart.data.csv' exists in your working directory
try:
    df = pd.read_csv('heart.data.csv')
    df = df.round(2)
    y_target_name = df.columns[-1]  # Assuming target is the last column
    y_data = df[y_target_name].values
    
    # Drop index/target columns for feature matrix
    X_df = df.drop(columns=[df.columns[0], y_target_name])
    feature_names = X_df.columns
    
    # Drop highly correlated features (>0.7) to prevent multicollinearity
    corr_matrix = X_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
    X_df = X_df.drop(columns=to_drop)
    
    print("\nFeatures retained after collinearity check:", list(X_df.columns))
    
    # Execute Model Pipeline
    model, model_name = get_regressor(1) # Select Linear Regression
    print(f"\nExecuting Pipeline with: {model_name}")
    
    best_r2 = 0
    best_features = None
    
    combinations_list = get_feature_combinations(len(X_df.columns), len(X_df.columns))
    
    for combo in combinations_list:
        X_subset = X_df.iloc[:, combo].values
        model.fit(X_subset, y_data)
        y_pred = model.predict(X_subset)
        
        r2 = r2_score(y_data, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_features = X_df.columns[combo].tolist()
            
    print(f"\nOptimal Feature Combination: {best_features}")
    print(f"Maximum R-squared achieved: {best_r2:.4f}")

except FileNotFoundError:
    print("\nDataset 'heart.data.csv' not found. Please ensure it is in the repository.")
