import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# --------------------------- Configuration ---------------------------
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="wide")

# --------------------------- Features ---------------------------
model_features = [
   'PhysicalHealthDays', 'MentalHealthDays', 'DrinksPerDay', 'Sex',
   'GeneralHealth', 'LastCheckupTime', 'PhysicalActivity',
   'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadOtherCancer',
   'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
   'HadDiabetes', 'DifficultyHearing', 'DifficultySeeing',
   'DifficultyMakingDecisions', 'DifficultyWalking', 'DifficultyDressing',
   'DifficultyErrands', 'SmokingStatus', 'AgeGroup', 'BMICategory',
   'EverTestedHIV', 'ReceivedFluVax', 'EverHadPneumoniaVax',
   'EverHadCOVID', 'HighBloodPressure', 'CholesterolCheck5yrs',
   'EverToldCHD', 'CalculatedCHD', 'HeavyDrinking', 'Smoked100Cigarettes',
   'CurrentlySmoke'
]

@st.cache_resource
def load_artifacts():
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    model = tf.keras.models.load_model("FNN.h5")
    return label_encoders, scaler, model

label_encoders, scaler, model = load_artifacts()

# --- Mapping from UI labels to the exact classes expected by the LabelEncoders ---

# Example: Your UI shows ['Male', 'Female'], but your LabelEncoder classes may be ['Female', 'Male']
# So to ensure proper encoding, build reverse lookup dicts for these inputs:

ui_to_le_mapping = {
    'Sex': {'Female': '0', 'Male': '1'},
    'GeneralHealth': {
        'Excellent': '0', 'Very Good': '1', 'Good': '2', 'Fair': '3', 'Poor': '4',
        'Unknown': '5', 'Refused': '6'  # Adjust if you have these extra classes
    },
    'LastCheckupTime': {
        '<1 year': '0', '1-2 years': '1', '2-5 years': '2', '5+ years': '3',
        'Never': '4', 'Refused': '5'  # Adjust if needed
    },
    'PhysicalActivity': {'Yes': '0', 'No': '1', 'Refused': '2', 'Unknown': '3'},
    'HadStroke': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadAsthma': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadSkinCancer': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadOtherCancer': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadCOPD': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadDepressiveDisorder': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadKidneyDisease': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadArthritis': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HadDiabetes': {'No': '0', 'Yes': '1', 'Borderline': '2', 'DuringPregnancy': '3', 'Refused': '4', 'Unknown': '5'},
    'DifficultyHearing': {'No': '0', 'Yes': '1', 'Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Deaf': '3'},

'DifficultySeeing': {'No': '0', 'Yes': '1', 'Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Blind': '3'},
    'DifficultyMakingDecisions': {'No': '0', 'Yes': '1', 'Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Cannot make decisions': '3'},
    'DifficultyWalking': {'No': '0',  'Yes': '1','Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Cannot walk': '3'},
    'DifficultyDressing': {'No': '0',  'Yes': '1','Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Cannot dress': '3'},
    'DifficultyErrands': {'No': '0',  'Yes': '1','Yes, some difficulty': '1', 'Yes, a lot of difficulty': '2', 'Cannot do errands': '3'},
    'SmokingStatus': {'Never smoked': '0','Never':'0','Former': '1', 'Current': '2', 'Refused': '3'},
    'AgeGroup': {
        '18-24': '0', '25-29': '1', '30-34': '2', '35-39': '3', '40-44': '4',
        '45-49': '5', '50-54': '6', '55-59': '7', '60-64': '8', '65-69': '9',
        '70-74': '10', '75-79': '11', '80+': '12'
    },
    'BMICategory': {'Underweight': '0', 'Normal weight': '1', 'Overweight': '2', 'Obese': '3'},
    'EverTestedHIV': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'ReceivedFluVax': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'EverHadPneumoniaVax': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'EverHadCOVID': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'HighBloodPressure': {'No': '0', 'Yes': '1'},
    'CholesterolCheck5yrs': {'No': '0', 'Yes': '1'},
    'EverToldCHD': {'No': '0', 'Yes': '1', 'Refused': '2', 'Unknown': '3'},
    'CalculatedCHD': {'No': '0', 'Yes': '1'},
    'HeavyDrinking': {'No': '0', 'Yes': '1'},
    'Smoked100Cigarettes': {'No': '0', 'Yes': '1'},
    'CurrentlySmoke': {'Never smoked': '0', 'Former smoker': '1', 'Current smoker': '2', 'Yes': '3', 'No': '4'}
}


def safe_transform(le, val, col):
    """
    Safely transform user input val using label encoder le.
    Maps UI label to LabelEncoder class if mapping exists.
    """
    # Map UI label to encoder label
    if col in ui_to_le_mapping:
        val_mapped = ui_to_le_mapping[col].get(val, None)
        if val_mapped is None:
            st.error(f"❌ Invalid input '{val}' for '{col}'")
            st.stop()
    else:
        val_mapped = val  # no mapping, use as is

    if val_mapped in le.classes_:
        return le.transform([val_mapped])[0]

    st.error(f"❌ Unseen label '{val_mapped}' for feature '{col}'")
    st.stop()

def encode_and_scale_input(input_dict):
    df = pd.DataFrame([{k: input_dict.get(k, np.nan) for k in model_features}])

    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda v: safe_transform(le, v, col))

    st.write("Encoded & ready for scaling input:")
    st.write(df)

    X_scaled = scaler.transform(df)
    return X_scaled


def predict_risk(input_dict):
    try:
        X = encode_and_scale_input(input_dict)
        prob = model.predict(X)[0][0]
        if prob < 0.33:
            risk = "Low"
        elif prob < 0.66:
            risk = "Medium"
        else:
            risk = "High"
        return prob, risk
    except Exception as e:
        return None, str(e)

# --------------------------- Layout with Images ---------------------------
left_col, center_col, right_col = st.columns([1, 4, 1])

with left_col:
    st.image("left_image.png", width=100)

with center_col:
    st.markdown("<h1 style='text-align: center;'>💓 Heart Attack Risk Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Fill out the following health information to estimate your heart attack risk.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><b>Contact:</b> <a href='mailto:emirt.worku99@email.com'>emirt.worku99@gmail.com</a></p>", unsafe_allow_html=True)

with right_col:
    st.image("right_image.png", width=100)

input_data = {}
col1, col2, col3, col4 = st.columns(4)

with col1:
    input_data['Sex'] = st.selectbox("Sex", ['Male', 'Female'])
    input_data['AgeGroup'] = st.selectbox("Age Group", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'])
    input_data['PhysicalHealthDays'] = st.number_input("Physical Health Days", 0, 30, 0)
    input_data['MentalHealthDays'] = st.number_input("Mental Health Days", 0, 30, 0)

with col2:
    input_data['DrinksPerDay'] = st.number_input("Drinks Per Day", 0.0, 20.0, 0.0)
    input_data['GeneralHealth'] = st.selectbox("General Health", ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
    input_data['LastCheckupTime'] = st.selectbox("Last Checkup Time", ['<1 year', '1-2 years', '2-5 years', '5+ years'])
    input_data['PhysicalActivity'] = st.selectbox("Physical Activity", ['Yes', 'No'])

with col3:
    input_data['BMICategory'] = st.selectbox("BMI Category", ['Underweight', 'Normal', 'Overweight', 'Obese'])
    input_data['SmokingStatus'] = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
    input_data['EverTestedHIV'] = st.selectbox("Ever Tested HIV", ['Yes', 'No'])

with col4:
    input_data['EverHadCOVID'] = st.selectbox("Ever Had COVID-19", ['Yes', 'No'])
    input_data['HighBloodPressure'] = st.selectbox("High Blood Pressure", ['Yes', 'No'])
    input_data['CholesterolCheck5yrs'] = st.selectbox("Cholesterol Check in 5 Yrs", ['Yes', 'No'])
    input_data['EverToldCHD'] = st.selectbox("Ever Told CHD", ['Yes', 'No'])

optional_fields = {
    'HadStroke': "Had Stroke?",
    'HadAsthma': "Had Asthma?",
    'HadSkinCancer': "Had Skin Cancer?",
    'HadOtherCancer': "Had Other Cancer?",
    'HadCOPD': "Had COPD?",
    'HadDepressiveDisorder': "Had Depressive Disorder?",
    'HadKidneyDisease': "Had Kidney Disease?",
    'HadArthritis': "Had Arthritis?",
    'HadDiabetes': "Had Diabetes?",
    'DifficultyHearing': "Difficulty Hearing?",
    'DifficultySeeing': "Difficulty Seeing?",
    'DifficultyMakingDecisions': "Difficulty Making Decisions?",
    'DifficultyWalking': "Difficulty Walking?",
    'DifficultyDressing': "Difficulty Dressing?",
    'DifficultyErrands': "Difficulty Doing Errands?",
    'ReceivedFluVax': "Received Flu Vaccine?",
    'EverHadPneumoniaVax': "Ever Had Pneumonia Vaccine?",
    'CalculatedCHD': "Calculated CHD?",
    'HeavyDrinking': "Heavy Drinking?",
    'Smoked100Cigarettes': "Smoked 100 Cigarettes?",
    'CurrentlySmoke': "Currently Smoke?"
}

with st.expander("Show Optional Fields"):
    opt_col1, opt_col2 = st.columns(2)
    
    for i, (key, label) in enumerate(optional_fields.items()):
        with (opt_col1 if i % 2 == 0 else opt_col2):
            if key in ui_to_le_mapping:
                options = list(ui_to_le_mapping[key].keys())
                input_data[key] = st.selectbox(label, options)
            else:
                input_data[key] = st.text_input(label, "")
if st.button("Predict Risk"):
    probability, risk_category = predict_risk(input_data)
    if probability is None:
        st.error(f"❌ Prediction error: {risk_category}")
    else:
        st.success(f"Risk Score: **{probability:.2f}**")
        st.markdown(f"Risk Level: **{risk_category.upper()}**")
        st.progress(min(int(probability * 100), 100))
