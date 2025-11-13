import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import os


def train(csv_fn, model_fn):
    """Train Random Forest regression model on dengue data."""
    print(f"[INFO] Loading training data from: {csv_fn}")
    df = pd.read_csv(csv_fn)

    features = ["rainfall", "mean_temperature"]
    target = "disease_cases"

    # Handle missing values
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"🌲 RandomForest MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")

    # Ensure model file ends with .bin
    if not model_fn.endswith(".bin"):
        model_fn += ".bin"

    # Ensure directory exists
    dir_path = os.path.dirname(model_fn)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    joblib.dump(model, model_fn)
    print(f"✅ Model saved to: {model_fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest regression model.")
    parser.add_argument("csv_fn", type=str, help="Path to CSV file for training data.")
    parser.add_argument("model_fn", type=str, help="Path to save trained model.")
    args = parser.parse_args()

    train(args.csv_fn, args.model_fn)
