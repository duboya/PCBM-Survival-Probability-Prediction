import streamlit as st
import pandas as pd
import joblib
import shap

# Define labels for survival time predictions
labels = ["year1_survival", "year3_survival", "year5_survival"]

# Define feature names and default values
feature_names = [
    "Liver Metastases", "T Stage", "Gleason Score", "Age at Diagnosis", "Months to Treatment",
    "Histological Type", "Surgery", "PSA", "Marital Status", "N Stage", "Grade", "Race",
    "Chemotherapy", "Brain Metastases", "Median Household Income", "Lung Metastases", "Radiotherapy"
]
default_values = {
    "Liver Metastases": 0, "T Stage": 3, "Gleason Score": 10, "Age at Diagnosis": 34,
    "Months to Treatment": 24, "Histological Type": 39, "Surgery": 0, "PSA": 187,
    "Marital Status": 1, "N Stage": 1, "Grade": 1, "Race": 3, "Chemotherapy": 0,
    "Brain Metastases": 0, "Median Household Income": 8, "Lung Metastases": 0, "Radiotherapy": 0
}

# Streamlit user interface title
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
        model = joblib.load(f'online_lr_model_{label}.pkl')

        # Make class prediction
        predicted_class = model.predict(features)[0]
        survival_status = "Surviving" if predicted_class == 1 else "Not Surviving"

        # Make probability prediction using predict_proba
        probabilities = model.predict_proba(features)
        probability_prediction = probabilities[0][1]

        prediction_results[label] = (survival_status, probability_prediction)

        try:
            # Use KernelExplainer if LinearExplainer fails
            explainer = shap.LinearExplainer(model, features, feature_dependence='independent')
        except ValueError:
            explainer = shap.KernelExplainer(model.predict_proba, features)

        # Calculate SHAP values
        shap_values = explainer.shap_values(features)

        # Generate SHAP force plot and convert to HTML
        shap_html = shap.force_plot(explainer.expected_value, shap_values[0], features.iloc[0],
                                     matplotlib=False, show=False)

        # Render the SHAP force plot HTML content in Streamlit
        st.components.v1.html(shap_html, height=400, width=1000)

    # Display prediction results
    st.write("### Prediction Results")
    for label, (survival_status, probability) in prediction_results.items():
        st.write(f"**{label}:** Predicted status = {survival_status}, Probability of survival = {probability:.2f}")
        probability_percent = probability * 100
        advice = (f"According to our model, you're predicted to have a probability of survival at "
                  f"{probability_percent:.1f}% for {label}." if survival_status == "Surviving" else
                  f"According to our model, you're predicted not to survive with a probability at "
                  f"{100 - probability_percent:.1f}% for {label}.")
        st.write(f"**Advice for {label}:** {advice}")
