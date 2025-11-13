import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# === 1. Load trained model ===
model_path = "output_forest/brazil_forest_model.bin"
model = joblib.load(model_path)
print(f"✅ Loaded model from {model_path}")

# === 2. Load dataset for prediction ===
data_path = "../exports/brazil.csv"
df = pd.read_csv(data_path)

if df.columns[0].startswith("Unnamed"):
    df.rename(columns={df.columns[0]: "id"}, inplace=True)

features = ["rainfall", "mean_temperature"]
target = "disease_cases"

# === 3. Prepare data ===
# For scoring, use only rows where target is available
mask_valid = ~df[target].isna()
X = df.loc[mask_valid, features].fillna(df[features].mean())
y_true = df.loc[mask_valid, target]

# === 4. Make predictions ===
y_pred = model.predict(X)
df.loc[mask_valid, "predicted_disease_cases"] = y_pred

# === 5. Evaluate performance ===
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# Safe MAPE (avoid division by zero)
mask_nonzero = y_true != 0
mape = mean_absolute_percentage_error(y_true[mask_nonzero], y_pred[mask_nonzero]) * 100 if mask_nonzero.any() else np.nan

print("\n📊 Random Forest Model Evaluation:")
print(f"MAE  (Mean Absolute Error):        {mae:.3f}")
print(f"RMSE (Root Mean Squared Error):    {rmse:.3f}")
print(f"R²   (Coefficient of Determination): {r2:.3f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

# === 6. Save predictions + report ===
predictions_path = "output_forest/brazil_predictions_forest.csv"
df.to_csv(predictions_path, index=False)
print(f"\n✅ Predictions saved to: {predictions_path}")



# === 7. Visualization ===

# Create output folder for plots
os.makedirs("output_forest/plots", exist_ok=True)

# === Predicted vs Actual ===
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="royalblue", edgecolor=None)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
plt.xlabel("Actual Disease Cases")
plt.ylabel("Predicted Disease Cases")
plt.title("Predicted vs Actual (Random Forest)")
plt.grid(True)
plt.tight_layout()

# 💾 Save figure
scatter_path = "output_forest/plots/predicted_vs_actual.png"
plt.savefig(scatter_path, dpi=300)
print(f"📈 Saved scatter plot to: {scatter_path}")
plt.show()

# === Feature Importance ===
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features, hue=features, palette="viridis", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

# 💾 Save figure
bar_path = "output_forest/plots/feature_importance.png"
plt.savefig(bar_path, dpi=300)
print(f"🌲 Saved feature importance plot to: {bar_path}")
plt.show()
# Optional: save a report with summary metrics
report_path = "output_forest/brazil_forest_report.txt"
with open(report_path, "w") as f:
    f.write(" Random Forest Regression Report \n")
    f.write(f"Data file: {data_path}\n")
    f.write(f"Model file: {model_path}\n")
    f.write(f"\nMAE : {mae:.3f}\n")
    f.write(f"RMSE: {rmse:.3f}\n")
    f.write(f"R²  : {r2:.3f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
print(f"📝 Report saved to: {report_path}")


# # === 8. Time-series visualization (smooth line version) ===
# if "time_period" in df.columns:
#     df_plot = df.loc[mask_valid, ["time_period", target, "predicted_disease_cases"]].copy()
#     df_plot["time_period"] = pd.to_datetime(df_plot["time_period"], errors="coerce")
#     df_plot = df_plot.sort_values("time_period")
#
#     plt.figure(figsize=(12, 5))
#     plt.plot(df_plot["time_period"], df_plot[target], label="Actual", color="black", linewidth=2)
#     plt.plot(df_plot["time_period"], df_plot["predicted_disease_cases"], label="Predicted", color="red", linewidth=2, alpha=0.8)
#     plt.title("Disease Cases Over Time (Brazil, Random Forest)")
#     plt.xlabel("Time Period")
#     plt.ylabel("Disease Cases")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#
#     timeplot_path = "output_forest/plots/time_series_comparison_line.png"
#     plt.savefig(timeplot_path, dpi=300)
#     print(f"📆 Saved smoothed time series plot to: {timeplot_path}")
#     plt.show()


# === Time-series line plot ===


if "time_period" in df.columns:
    # Kopier og konverter dato
    df_plot = df.loc[mask_valid, ["time_period", target, "predicted_disease_cases"]].copy()
    df_plot["time_period"] = pd.to_datetime(df_plot["time_period"], errors="coerce")

    # 🔹 Gruppér per måned (for å få jevne linjer)
    df_plot = (
        df_plot.groupby(df_plot["time_period"].dt.to_period("M"))
        [["disease_cases", "predicted_disease_cases"]]
        .mean()
        .reset_index()
    )
    df_plot["time_period"] = df_plot["time_period"].dt.to_timestamp()

    # 🔹 Tegn linjeplot uten skygge eller søyler
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot["time_period"], df_plot["disease_cases"],
             label="Actual", color="black", linewidth=2)
    plt.plot(df_plot["time_period"], df_plot["predicted_disease_cases"],
             label="Predicted", color="red", linewidth=2, alpha=0.8)
    plt.title("Monthly Disease Cases (Brazil, Random Forest)")
    plt.xlabel("Time Period")
    plt.ylabel("Average Disease Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Lagre og vise
    timeplot_path = "output_forest/plots/time_series_monthly_lineplot.png"
    plt.savefig(timeplot_path, dpi=300)
    print(f"📆 Saved monthly smoothed line plot to: {timeplot_path}")
    plt.show()


