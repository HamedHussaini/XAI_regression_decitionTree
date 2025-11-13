# === lime_explain_brazil.py ===
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import os

# === 1. Load trained model and data ===
model = joblib.load("output/brazil_model.bin")
df = pd.read_csv("./example_data/historic_brazil.csv")

features = ["rainfall", "mean_temperature"]
target = "disease_cases"

# Drop missing values and select features
df = df.dropna(subset=features + [target])
X = df[features].values
y = df[target].values

# === 2. Initialize LIME explainer ===
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=features,
    mode="regression"
)

# === 3. Choose one instance to explain ===
instance_index = 100  # you can change this
instance = X[instance_index]

print(f"Explaining instance #{instance_index}:")
print(pd.Series(instance, index=features))

# === 4. Generate explanation ===
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict,
    num_features=len(features)
)

# === 5. Print explanation in console ===
print("\n LIME Explanation (feature contribution):")
for feature, weight in exp.as_list():
    print(f"{feature:25s} {weight:+.3f}")

# === 6. Visualize and save ===
os.makedirs("lime_brazil", exist_ok=True)
fig = exp.as_pyplot_figure()
plt.title(f"LIME Local Explanation – Instance {instance_index}")
plt.tight_layout()
plt.savefig("lime_brazil/lime_local.png", dpi=300)
plt.show()

print("\n LIME explanation saved to lime_brazil/lime_local.png")
