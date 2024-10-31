import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options
T_options = {    
    0: 'I (0)',    
    1: 'II (1)',    
    2: 'III (2)',    
    3: 'IV (3)'
}


# Define feature names
feature_names = [    
    "Age", "Tumor size", "Distant lymph node metastasis", "EOD T", "EOD N",    
    "Brain metastasis", "Lung metastasis", "Radiotherapy", "Chemotherapy"
]

# Streamlit user interface
st.title("Predicted risk of bone metastasis from hepatocellular carcinoma")

# age: categorical selection
Age = st.selectbox("Age (0=<60, 1=>60):", options=[0, 1], format_func=lambda x: '<60 (0)' if x == 0 else '>60 (1)')

# Tumor_Size: numerical input
Tumor_Size = st.number_input("Tumor size:", min_value=2, max_value=280, value=50)

# Distant_lymph: categorical selection
Distant_lymph = st.selectbox("Distant lymph node metastasis (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# T: categorical selection
T = st.selectbox("EOD T:", options=list(T_options.keys()), format_func=lambda x: T_options[x])

# N: categorical selection
N = st.selectbox("EOD N (0=N0, 1=N1):", options=[0, 1], format_func=lambda x: 'N0 (0)' if x == 0 else 'N1 (1)')

# Brain_metastasis: categorical selection
Brain_metastasis = st.selectbox("Brain metastasis (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# lung_metastasis: categorical selection
lung_metastasis = st.selectbox("lung metastasis (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Radiotherapy: categorical selection
Radiotherapy = st.selectbox("Radiotherapy (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Chemotherapy: categorical selection
Chemotherapy = st.selectbox("Chemotherapy (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')


# Process inputs and make predictions
feature_values = [Age, Tumor_Size, Distant_lymph, T, N, Brain_metastasis, lung_metastasis, Radiotherapy, Chemotherapy]
features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:        
        advice = (            
            f"Our model indicates that you have a high risk of bone metastasis from hepatocellular carcinoma. "
            f"The predicted probability of developing bone metastasis is {probability:.1f}%. "
            "While this is an estimate, it suggests that you may be at significant risk. "
            "I recommend consulting an oncologist as soon as possible for further evaluation, "
            "to ensure accurate diagnosis and appropriate treatment."       
        )    
    else:        
         advice = (            
            f"Our model indicates that you have a low risk of bone metastasis from hepatocellular carcinoma. "
            f"The predicted probability of not having bone metastasis is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle remains essential. "
            "I recommend regular check-ups to monitor your health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )
    st.write(advice)

    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200) 

    st.image("shap_force_plot.png")