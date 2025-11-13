import joblib
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# === 1. Load trained model ===
model = joblib.load("output_forest/brazil_forest_model.bin")

# === 2. Load dataset ===
df = pd.read_csv("../exports/brazil.csv")

# === 3. Prepare data ===
features = ["rainfall", "mean_temperature"]
X = df[features].fillna(df[features].mean())


# === 4. Create PDP plots ===
features_to_plot = ["rainfall", "mean_temperature"]

# Opprett figur først
fig, ax = plt.subplots(figsize=(10, 4))

PartialDependenceDisplay.from_estimator(
    model,
    X,
    features=features_to_plot,
    kind="average",
    grid_resolution=50,
    ax=ax
)

plt.suptitle("Partial Dependence of Rainfall and Temperature on Disease Cases")
plt.tight_layout()
plt.savefig("output_forest/xai/pdp_brazil.png", dpi=300)
plt.show()
