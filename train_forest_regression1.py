import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# === 1. Load dataset ===
df = pd.read_csv("../exports/brazil.csv")

# Fix unnamed index column
if df.columns[0].startswith("Unnamed"):
    df.rename(columns={df.columns[0]: "id"}, inplace=True)

print("\n✅ Loaded data from exports/brazil.csv")
print("Columns:", df.columns.tolist())

# === 2. Define features and target ===
features = ["rainfall", "mean_temperature"]
target = "disease_cases"

# Check required columns
missing_cols = [c for c in features + [target] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

X = df[features]
y = df[target]

# === 3. Handle missing values ===
mask_valid = ~y.isna()
X = X.loc[mask_valid].fillna(X.mean())
y = y.loc[mask_valid]

print(f"✅ Dataset size after cleaning: {len(X)}")

# === 4. Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Train Random Forest model ===
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# === 6. Evaluate model ===
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

# Handle MAPE safely
mask_nonzero = y_val != 0
if mask_nonzero.sum() > 0:
    mape = mean_absolute_percentage_error(y_val[mask_nonzero], y_pred[mask_nonzero]) * 100
else:
    mape = np.nan

print("\n🌲 Random Forest Evaluation:")
print(f"MAE : {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")
print(f"MAPE: {mape:.2f}%")

# === 7. Feature importance ===
print("\nFeature importances:")
for f, imp in zip(features, model.feature_importances_):
    print(f"  {f}: {imp:.4f}")

# === 8. Save model ===
joblib.dump(model, "output_forest/brazil_forest_model.bin")
print("\n✅ Model saved to brazil_forest_model.bin")
