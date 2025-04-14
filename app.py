import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

mandatory_features = [
    'Sex', 'AgeCategory', 'SystolicBP', 'DiastolicBP', 'Cholesterol',
    'BMI', 'SmokerStatus', 'AlcoholDrinkers', 'HadDiabetes',
    'HadAngina', 'HadStroke', 'PhysicalActivities', 'PrematureFamilyHistory'
]

model_features = [
    'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'PhysicalActivities', 'SleepHours',
    'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
    'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'SmokerStatus', 'ChestScan', 'AgeCategory', 'BMI',
    'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos', 'SystolicBP',
    'DiastolicBP', 'Cholesterol', 'PrematureFamilyHistory'
]

categorical_inputs = [
    'Sex', 'GeneralHealth', 'SmokerStatus', 'AlcoholDrinkers',
    'ChestScan', 'AgeCategory', 'HIVTesting', 'FluVaxLast12',
    'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos',
    'LastCheckupTime'
]
def preprocess_input(input_dict, feature_list):
    df = pd.DataFrame([input_dict])[feature_list]
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def predict_risk(input_data):
    model_path = "heart_attack_model.h5"
    try:
        input_processed = preprocess_input(input_data, model_features)
    except KeyError as e:
        return None, f"⚠️ Missing required feature: {str(e)}"

    model = tf.keras.models.load_model(model_path)
    probability = model.predict(input_processed)[0][0]

    if probability < 0.33:
        risk = "Low"
    elif probability < 0.66:
        risk = "Medium"
    else:
        risk = "High"

    return probability, risk
# Start of your GUI


st.set_page_config(page_title="CVDs Risk predictor", layout="wide")
st.title("💓 CVDs Risk Predictor")
st.markdown("Fill out the following health information to estimate your heart attack risk.")
# Custom layout with two images on left and right, and centered title/email
left_col, center_col, right_col = st.columns([1, 4, 1])

with left_col:
    st.image("left_image.png", width=100)

with center_col:
    st.markdown("<h1 style='text-align: center;'>💓 CVDs Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Fill out the following health information to estimate your heart attack risk.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><b>Contact:</b> <a href='mailto:emirt.worku99@email.com'>emirt.worku99@gmail.com</a></p>", unsafe_allow_html=True)

with right_col:
    st.image("right_image.png", width=100)
# Mandatory features — Column 1 and 2
col1, col2, col3, col4 = st.columns(4)
input_data = {}
with col1:
    input_data['Sex'] = st.selectbox("Sex", ['Male', 'Female'])
    input_data['AgeCategory'] = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'
    ])
    input_data['SystolicBP'] = st.number_input("Systolic Blood Pressure", 80, 250, 120)
    input_data['DiastolicBP'] = st.number_input("Diastolic Blood Pressure", 40, 150, 80)

with col2:
    input_data['Cholesterol'] = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    input_data['BMI'] = st.number_input("Body Mass Index", 10.0, 60.0, 24.0)
    input_data['SmokerStatus'] = st.selectbox("smoker status", ['Former smoker' ,'Never smoked' ,'Current smoker - now smokes every day',
 'Current smoker - now smokes some days'])
    input_data['AlcoholDrinkers'] = st.selectbox("Do you drink alcohol?", ['Yes', 'No'])

with col3:
    input_data['HadDiabetes'] = st.selectbox("Do you have diabetes?", ['Yes', 'No'])
    input_data['HadAngina'] = st.selectbox("Have you had angina?", ['Yes', 'No'])
    input_data['HadStroke'] = st.selectbox("Have you had a stroke?", ['Yes', 'No'])
    input_data['PhysicalActivities'] = st.selectbox("Do you exercise regularly?", ['Yes', 'No'])

with col4:
    input_data['PrematureFamilyHistory'] = st.selectbox("Family history of early heart disease?", ['Yes', 'No'])

    # ✅ Optional Fields
    input_data['GeneralHealth'] = st.selectbox("General Health", ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    input_data['MentalHealthDays'] = st.number_input("Mental Health (days/month)", 0, 30, 0)
    input_data['PhysicalHealthDays'] = st.number_input("Physical Health (days/month)", 0, 30, 0)
    input_data['SleepHours'] = st.number_input("Average Sleep Hours", 0, 24, 7)
    input_data['LastCheckupTime'] = st.selectbox("LastCheckupTime", [ 'Within past year (anytime less than 12 months ago)',
    'Within past 2 years (1 year but less than 2 years ago)','Within past 5 years (2 years but less than 5 years ago)','5 or more years ago'])
# More optional inputs (added here but not mandatory)

optional_fields = {
    'HadAsthma': "Do you have asthma?",
    'HadSkinCancer': "Do you have skin cancer?",
    'HadCOPD': "Do you have COPD?",
    'HadDepressiveDisorder': "Depressive disorder?",
    'HadKidneyDisease': "Kidney disease?",
    'HadArthritis': "Arthritis?",
    'DeafOrHardOfHearing': "Deaf or hearing difficulty?",
    'BlindOrVisionDifficulty': "Vision difficulty?",
    'DifficultyConcentrating': "Difficulty concentrating?",
    'DifficultyWalking': "Difficulty walking?",
    'DifficultyDressingBathing': "Difficulty dressing/bathing?",
    'DifficultyErrands': "Difficulty doing errands?",
    'ChestScan': "Had chest scan recently?",
    'HIVTesting': "Ever tested for HIV?",
    'FluVaxLast12': "Got flu vaccine in past year?",
    'PneumoVaxEver': "Ever had pneumococcal vaccine?",
    'TetanusLast10Tdap': "Had Tetanus/Tdap in past 10 yrs?",
    'HighRiskLastYear': "High risk behavior last year?",
    'CovidPos': "Ever tested COVID-19 positive?"
}
with st.expander("➕ Show Optional Fields"):
    opt_col1, opt_col2 = st.columns(2)
    for i, (key, label) in enumerate(optional_fields.items()):
        with (opt_col1 if i % 2 == 0 else opt_col2):
            input_data[key] = st.selectbox(label, ['Yes', 'No'])

# Predict button
if st.button("🧠 Predict Risk"):
    probability, risk_category = predict_risk(input_data)

    if probability is None:
        st.error(risk_category)
    else:
        st.success(f"Risk Score: **{probability:.2f}**")
        st.markdown(f"### 🚦 Risk Level: **{risk_category.upper()}**")
        st.progress(min(int(probability * 100), 100))
