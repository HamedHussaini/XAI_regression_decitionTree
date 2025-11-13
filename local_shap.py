import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# === 1. Last modell og data ===
model = joblib.load("output/brazil_model.bin")
df = pd.read_csv("./example_data/historic_brazil.csv")

features = ["rainfall", "mean_temperature"]
df = df.dropna(subset=features)
X = df[features].astype(float)

# === 2. Opprett mappe for resultater ===
os.makedirs("shap_brazil/local", exist_ok=True)

# === 3. Sett opp TreeExplainer med robust parametre ===
# Bruker en liten bakgrunnsprøve for stabilitet
explainer = shap.TreeExplainer(
    model,
    data=X.sample(min(200, len(X)), random_state=0),
    feature_perturbation="interventional"
)

# Merk: KALL shap_values() direkte for å unngå additivity-feil
shap_values = explainer.shap_values(X, check_additivity=False)

# === 4A. Waterfall-plot ===
index = 100
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X.iloc[index],
        feature_names=features,
    ),
    show=False,
)
plt.title(f"Local SHAP Waterfall – Instance {index}")
plt.tight_layout()
plt.savefig("shap_brazil/local/waterfall_local.png", dpi=300, bbox_inches="tight")
plt.show()

# === 4B. Bar-plot ===
plt.figure()
shap.plots.bar(
    shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X.iloc[index],
        feature_names=features,
    ),
    show=False,
)
plt.title(f"Local SHAP Bar Plot – Instance {index}")
plt.tight_layout()
plt.savefig("shap_brazil/local/bar_local.png", dpi=300, bbox_inches="tight")
plt.show()

print(" Lagret som waterfall_local.png og bar_local.png i shap_brazil/")
