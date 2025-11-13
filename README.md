# Dengue Forecasting with Explainable AI (XAI)

This project predicts **dengue disease cases** in Brazil using a **Random Forest Regression model**, and explains its predictions with **model-agnostic XAI methods** such as **SHAP** and **LIME**.

It demonstrates how explainable machine learning can help uncover relationships between **climate variables (rainfall, temperature)** and **health outcomes (dengue incidence)** — bridging AI and public health insight.

---

## Overview

| Component | Description |
|------------|-------------|
| **Model Type** | Random Forest Regressor (scikit-learn) |
| **XAI Methods** | SHAP (global + local), LIME (local), Partial Dependence, Feature Importance |
| **Data Source** | Historical climate and dengue case data (Brazil, CHAP-compatible format) |
| **Goal** | Forecast dengue outbreaks and interpret key climate drivers |

---

## Project Structure

XAI_regression_decisionTree/
│
├── example_data/
│ ├── historic_brazil.csv # historical training data
│ └── future_brazil.csv # future climate data for prediction
│
├── output/
│ ├── brazil_model.bin # trained model (binary)
│ └── predictions.csv # predicted dengue cases
│
├── shap_brazil/ # SHAP explanations
│ ├── global/ # global feature importance plots
│ └── local/ # local instance explanations
│
├── lime_brazil/ # LIME explanations
│
├── train.py # trains Random Forest model
├── predict.py # runs predictions using trained model
├── XAI_brazil_forest_shap.py # local SHAP explanations
├── XAI_brazil_forest_global.py # global SHAP analysis
├── lime_explain_brazil.py # LIME local explanation script
├── isolated_run.py # test script (train + predict pipeline)
├── requirements.txt # dependencies
└── README.md # project documentation

## Usage
### 1️⃣ Train the model
python train.py example_data/historic_brazil.csv output/brazil_model.bin

### 2️⃣ Generate predictions
python predict.py output/brazil_model.bin example_data/historic_brazil.csv example_data/future_brazil.csv output/predictions.csv

### 3️⃣ Explain the model with SHAP
Local explanation (one instance)
```
python XAI_brazil_forest_shap.py
```
Global explanation (all data)
```
python XAI_brazil_forest_global.py
```
### 4️⃣ Explain with LIME
```
python lime_explain_brazil.py
```
## 🧩 Explainable AI Methods
Method	Type	Purpose	Description
SHAP (SHapley Additive exPlanations)	Model-agnostic / Global + Local	Quantifies how each feature contributes to each prediction, ensuring consistency and additivity.	
LIME (Local Interpretable Model-agnostic Explanations)	Model-agnostic / Local	Approximates the complex model locally around one instance using a simple, interpretable model.	
Permutation Importance	Model-agnostic / Global	Measures how performance changes when a feature is randomly shuffled.	
Partial Dependence Plot (PDP)	Model-agnostic / Global	Shows how predicted outcomes change as one or two features vary.	
## Example Outputs

### SHAP Summary (Global)

Shows which climate features most influence dengue predictions.

### SHAP Dependence Plot

Visualizes interaction between rainfall and mean temperature.

### LIME Local Explanation

Explains why the model predicted a specific value.

## Interpretation Highlights

Rainfall generally increases predicted dengue cases when above seasonal averages.

Mean temperature influences dengue transmission risk non-linearly (moderate temps amplify mosquito activity).

Model explanations allow local health authorities to prioritize surveillance and intervention based on climate forecasts.

## Technologies Used
Category	Tools
Programming	Python 3.10+
ML Framework	scikit-learn
Explainability	SHAP, LIME
Visualization	matplotlib
Data Handling	pandas, numpy 

### Research Relevance

This project demonstrates interpretable forecasting in the climate-health domain.
It aligns with the CHAP (Climate and Health Analytics Platform) initiative and provides a reproducible framework for explainable regression models in epidemiological applications.