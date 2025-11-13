# === XAI_brazil_forest_shap.py ===
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# === 1. Last modell og data ===
model = joblib.load("output/brazil_model.bin")
df = pd.read_csv("./example_data/historic_brazil.csv")

features = ["rainfall", "mean_temperature"]
X = df[features].fillna(df[features].mean())

# === 2. Sett opp SHAP-explainer ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === 3. Opprett mappe for resultater ===
os.makedirs("shap_brazil", exist_ok=True)

# === 4. Global forklaring (summary plot) ===
shap.summary_plot(shap_values, X, feature_names=features, show=False)
plt.title("SHAP Summary Plot (Brazil Random Forest)")
plt.tight_layout()
plt.savefig("shap_brazil/shap_summary_brazil.png", dpi=300)
plt.show()

# === 5. Lokal forklaring (lagres som PNG) ===
index = 100
sample = X.iloc[[index]]

# Hvis shap_values er objekt med .values-attributt, hent det
values = shap_values.values if hasattr(shap_values, "values") else shap_values

# Bruk Matplotlib-modus for å kunne lagre som bilde
shap.force_plot(
    explainer.expected_value,
    values[index, :],
    sample,
    matplotlib=True
)

plt.title(f"Local SHAP Force Plot – Instance {index}")
plt.savefig("shap_brazil/force_plot_example.png", dpi=300, bbox_inches="tight")
plt.show()
