import argparse
import joblib
import pandas as pd
import os


def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    """Predict dengue cases using a trained Random Forest model (CHAP-compatible)."""
    print(f"[INFO] Loading model from: {model_fn}")
    if not os.path.exists(model_fn) and os.path.exists(model_fn + ".bin"):
        model_fn += ".bin"

    model = joblib.load(model_fn)

    # Load future data
    print(f"[INFO] Loading future data from: {future_climatedata_fn}")
    df_future = pd.read_csv(future_climatedata_fn)

    required_columns = ["time_period", "rainfall", "mean_temperature", "location"]
    missing = [c for c in required_columns if c not in df_future.columns]
    if missing:
        raise ValueError(f" Missing required columns: {missing}")

    # Prepare features
    X_future = df_future[["rainfall", "mean_temperature"]].fillna(df_future.mean(numeric_only=True))

    # Predict
    preds = model.predict(X_future)

    # CHAP-compatible output
    df_pred = pd.DataFrame({
        "time_period": df_future["time_period"],
        "location": df_future["location"],
        "sample_0": preds.astype(float)
    })

    # Save predictions
    out_dir = os.path.dirname(predictions_fn)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_pred.to_csv(predictions_fn, index=False)
    print(f" Predictions saved to: {predictions_fn}")
    print(df_pred.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict dengue cases using a trained model (CHAP-compatible).")
    parser.add_argument("model_fn", type=str, help="Path to trained model file.")
    parser.add_argument("historic_data_fn", type=str, help="Path to historic data CSV (used for compatibility).")
    parser.add_argument("future_climatedata_fn", type=str, help="Path to future climate data CSV.")
    parser.add_argument("predictions_fn", type=str, help="Path to save CHAP-compatible predictions CSV.")
    args = parser.parse_args()

    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
