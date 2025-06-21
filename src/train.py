import numpy as np
import pandas as pd
import pickle
from utils import scale_features
import os

def compute_cost(w, X, y, b):
    m = X.shape[0]
    f_wb = X @ w + b
    return (1/(2*m)) * np.sum((f_wb - y) ** 2)

def compute_gradient(w, X, y, b):
    m = X.shape[0]
    predictions = X @ w + b
    error = predictions - y
    dj_dw = (1 / m) * (X.T @ error)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = w_in.copy()
    b = b_in
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(w, X, y, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(w, X, y, b)
        cost_history.append(cost)
        if i % 1000 == 0 or i == num_iters - 1:
            print(f"Iteration {i}: Cost {cost:.4f}")
    return w, b, cost_history

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def main():
    # Construct paths relative to the script's location
    script_dir = os.path.dirname(__file__)
    root_dir = os.path.join(script_dir, '..')
    
    data_path = os.path.join(root_dir, 'data', 'final_jallianwala_dataset.csv')
    model_path = os.path.join(root_dir, 'models', 'jallianwala_model.pkl')
    scaler_path = os.path.join(root_dir, 'models', 'scaler.pkl')

    df = pd.read_csv(data_path)
    feature_cols = ['NewsArticles', 'YouTubeUploads', 'WeightedEvents', 'InverseDays', 'DaysSquared']
    target_col = 'SearchInterest'
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_ratio = 0.8
    train_size = int(len(df_shuffled) * train_ratio)
    train_set = df_shuffled.iloc[:train_size]
    test_set = df_shuffled.iloc[train_size:]
    X_train = train_set[feature_cols].values
    y_train = train_set[target_col].values
    X_test = test_set[feature_cols].values
    y_test = test_set[target_col].values
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_scaled = scale_features(X_train, X_mean, X_std)
    X_test_scaled = scale_features(X_test, X_mean, X_std)
    m, n = X_train_scaled.shape
    w_init = np.zeros(n)
    b_init = 0
    alpha = 0.02
    iterations = 100000
    w_final, b_final, cost_hist = gradient_descent(X_train_scaled, y_train, w_init, b_init, alpha, iterations)
    y_pred = X_test_scaled @ w_final + b_final
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    # Save model
    model_params = {'weights': w_final, 'bias': b_final}
    with open(model_path, "wb") as f:
        pickle.dump(model_params, f)
    scaler = {'mean': X_mean.tolist(), 'std': X_std.tolist()}
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"✅ Model and scaler saved to {os.path.relpath(os.path.dirname(model_path))}/ directory.")

if __name__ == "__main__":
    main() 