import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

# Load the model
labels = ["year1_survival", "year3_survival", "year5_survival"]

# Define feature names and default values
feature_names = [
    "Liver Metastases", "T Stage", "Gleason Score", "Age at Diagnosis", "Months to Treatment",
    "Histological Type", "Surgery", "PSA", "Marital Status", "N Stage", "Grade", "Race", 
    "Chemotherapy", "Brain Metastases", "Median Household Income", "Lung Metastases", "Radiotherapy"
]
default_values = {
    "Liver Metastases": 2, "T Stage": 0, "Gleason Score": 1, "Age at Diagnosis": 20, 
    "Months to Treatment": 0, "Histological Type": 39, "Surgery": 0, "PSA": 973,
    "Marital Status": 0, "N Stage": 3, "Grade": 0, "Race": 3, "Chemotherapy": 0,
    "Brain Metastases": 0, "Median Household Income": 8, "Lung Metastases": 2, "Radiotherapy": 2
}

# Define a function to turn log odds into probabilities
def logit_to_prob(logit):
    odds = np.exp(logit)
    return odds / (1 + odds)

# Streamlit user interface
st.title("PCBM Survival Probability Prediction")

# Get user inputs
user_inputs = {}
for feature, default_value in default_values.items():
    user_inputs[feature] = st.number_input(feature, value=default_value)

# Prepare the input for prediction
features = pd.DataFrame([list(user_inputs.values())], columns=feature_names)

# Initialize prediction results
prediction_results = {}

# Process the button click for prediction
if st.button("Predict Survival"):
    for label in labels:
        # Load the model using joblib
        model = joblib.load(f'xgb_model_{label}.pkl')

        # Convert the DataFrame to DMatrix
        dmatrix = xgb.DMatrix(features)
        
        # Make prediction
        logit_prediction = model.predict(dmatrix)
        probability_prediction = logit_to_prob(logit_prediction)[0]
        
        # Store the prediction result
        prediction_results[label] = probability_prediction
    
    # Display prediction results
    st.write("### Prediction Results")
    for label, probability in prediction_results.items():
        st.write(f"**{label}:** {probability:.2f}")

    # SHAP analysis
    for label in labels:
        # For SHAP, use the loaded model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # Visualize the first prediction's explanation
        st.write(f"### SHAP Force Plot for {label}")
        shap.force_plot(explainer.expected_value, shap_values[0], features, matplotlib=True)
        plt.savefig(f"shap_force_plot_{label}.png", bbox_inches='tight', dpi=1200)
        st.image(f"shap_force_plot_{label}.png")
