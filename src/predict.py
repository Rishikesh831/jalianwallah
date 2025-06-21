import numpy as np
import sys
import os
from utils import load_model, load_scaler, scale_features

FEATURES = ['NewsArticles', 'YouTubeUploads', 'WeightedEvents', 'InverseDays', 'DaysSquared']

# Construct paths relative to the script's location
script_dir = os.path.dirname(__file__)
root_dir = os.path.join(script_dir, '..')
MODEL_PATH = os.path.join(root_dir, 'models', 'jallianwala_model.pkl')
SCALER_PATH = os.path.join(root_dir, 'models', 'scaler.pkl')

def predict(features):
    w, b = load_model(MODEL_PATH)
    mean, std = load_scaler(SCALER_PATH)
    X = np.array(features, dtype=float)
    X_scaled = scale_features(X, mean, std)
    pred = X_scaled @ w + b
    return pred

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(f"Usage: python predict.py <NewsArticles> <YouTubeUploads> <WeightedEvents> <InverseDays> <DaysSquared>")
        sys.exit(1)
    features = [float(x) for x in sys.argv[1:]]
    result = predict(features)
    print(f"Predicted Public Interest: {result:.2f}") 