import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# === 1. Load model and data ===
model = joblib.load("output/brazil_model.bin")
df = pd.read_csv("example_data/historic_brazil.csv")

features = ["rainfall", "mean_temperature"]
df = df.dropna(subset=features)
X = df[features].astype(float)

# === 2. Prepare output folder ===
os.makedirs("shap_brazil/global", exist_ok=True)

# === 3. Build TreeExplainer ===
explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
shap_values = explainer.shap_values(X, check_additivity=False)

# === 4. Global summary plot (beeswarm) ===
plt.figure()
shap.summary_plot(shap_values, X, feature_names=features, show=False)
plt.title("Global SHAP Summary (All Instances)")
plt.tight_layout()
plt.savefig("shap_brazil/global/global_summary.png", dpi=300, bbox_inches="tight")
plt.show()

# === 5. Global feature importance (mean(|SHAP|)) ===
plt.figure()
shap.summary_plot(shap_values, X, feature_names=features, plot_type="bar", show=False)
plt.title("Global Feature Importance (Mean |SHAP|)")
plt.tight_layout()
plt.savefig("shap_brazil/global/global_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# === 6. Dependence plots (effect of one feature vs. its SHAP value) ===
for feat in features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X, show=False)
    plt.title(f"SHAP Dependence Plot – {feat}")
    plt.tight_layout()
    plt.savefig(f"shap_brazil/global/dependence_{feat}.png", dpi=300, bbox_inches="tight")
    plt.show()

# === 7. Violin-style distribution (alternative visualization) ===
plt.figure()
shap.plots.violin(shap.Explanation(values=shap_values, data=X, feature_names=features), show=False)
plt.title("SHAP Value Distributions (Violin Plot)")
plt.tight_layout()
plt.savefig("shap_brazil/global/global_violin.png", dpi=300, bbox_inches="tight")
plt.show()

print("All global SHAP plots saved to /shap_brazil/global:")
print("- global_summary.png")
print("- global_importance.png")
print("- dependence_rainfall.png, dependence_mean_temperature.png")
print("- global_violin.png")
