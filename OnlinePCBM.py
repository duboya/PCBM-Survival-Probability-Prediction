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
    "Liver Metastases": 0, "T Stage": 10, "Gleason Score": 10, "Age at Diagnosis": 34, 
    "Months to Treatment": 24, "Histological Type": 39, "Surgery": 0, "PSA": 187,
    "Marital Status": 1, "N Stage": 1, "Grade": 1, "Race": 3, "Chemotherapy": 0,
    "Brain Metastases": 0, "Median Household Income": 8, "Lung Metastases": 0, "Radiotherapy": 0
}

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
        model = joblib.load(f'online_xgb_model_{label}.pkl')
        
        # Make class prediction
        predicted_class = model.predict(features)[0]
        
        # Make probability prediction using predict_proba
        probabilities = model.predict_proba(features)
        probability_prediction = probabilities[0][1]  # Probability of survival
        
        # Store the prediction result
        prediction_results[label] = (predicted_class, probability_prediction)
        
    # Display prediction results
    st.write("### Prediction Results")
    for label, (predicted_class, probability) in prediction_results.items():
        st.write(f"**{label}:** Predicted class = {predicted_class}, Probability of survival = {probability:.2f}")

        # Generate advice based on prediction results
        probability_percent = probability * 100
        if predicted_class == 1:
            advice = (
                f"According to our model, you have a higher probability of survival. "
                f"The model predicts that your probability of surviving {label} is {probability_percent:.1f}%. "
                "It's important to continue following your treatment plan and have regular consultations with your oncologist."
            )
        else:
            advice = (
                f"According to our model, you have a lower probability of survival. "
                f"The model predicts that your probability of not surviving {label} is {100 - probability_percent:.1f}%. "
                "We recommend discussing additional treatment options with your healthcare provider and maintaining ongoing follow-up."
            )
        
        st.write(f"**Advice for {label}:** {advice}")
    
    # SHAP analysis
    for label in labels:
        # Reload model for SHAP calculation
        model = joblib.load(f'online_xgb_model_{label}.pkl')
        explainer = shap.Explainer(model)
        shap_values = explainer(features)
        # Visualize the first prediction's explanation
        st.write(f"### SHAP Force Plot for {label}")
        shap.force_plot(explainer.expected_value, shap_values.values[0], features, matplotlib=True, show=False)
        plt.savefig(f"shap_force_plot_{label}.png", bbox_inches='tight', dpi=1200)
        st.image(f"shap_force_plot_{label}.png")
