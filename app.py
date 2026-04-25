import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title('Factors Contributing to Cancer Risk')
st.markdown("This app predicts a user's Overall Risk Score of Cancer given lifestyle and genetic factors.")

# ============================================================
# SIDEBAR INPUT CONTROLS (12 features only)
# ============================================================

st.sidebar.header("Lifestyle & Health Factors")

# Core risk factors (matching your 12 features)
air_pollution = st.sidebar.slider('Air Pollution Exposure (0-10)', 0.0, 10.0, 5.0,
                                   help="0 = Clean air, 10 = High pollution area")

smoking = st.sidebar.slider('Smoking Level (0-10)', 0.0, 10.0, 5.0, 
                             help="0 = Non-smoker, 10 = Heavy smoker")

alcohol_use = st.sidebar.slider('Alcohol Use (0-10)', 0.0, 10.0, 5.0,
                                 help="0 = No alcohol, 10 = Heavy alcohol consumption")

diet_salted_processed = st.sidebar.slider('Salted/Processed Foods (0-10)', 0.0, 10.0, 5.0,
                                           help="0 = No processed foods, 10 = High intake")

obesity = st.sidebar.slider('Obesity Level (0-10)', 0.0, 10.0, 5.0,
                             help="0 = Normal weight, 10 = Severely obese")

diet_red_meat = st.sidebar.slider('Red Meat Consumption (0-10)', 0.0, 10.0, 5.0,
                                   help="0 = No red meat, 10 = Daily high consumption")

age = st.sidebar.slider('Age', 0, 100, 50)

bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)

family_history = st.sidebar.selectbox('Family History of Cancer', ['No', 'Yes'])
family_history_encoded = 1 if family_history == "Yes" else 0

fruit_veg_intake = st.sidebar.slider('Fruit & Vegetable Intake (0-10)', 0.0, 10.0, 5.0,
                                      help="0 = No fruits/vegetables, 10 = High intake")

physical_inactivity = st.sidebar.slider('Physical Inactivity Level (0-10)', 0.0, 10.0, 5.0,
                                         help="0 = Very active, 10 = Sedentary")

brca_mutation = st.sidebar.selectbox('BRCA Gene Mutation', ['No', 'Yes'])
brca_mutation_encoded = 1 if brca_mutation == "Yes" else 0

# ============================================================
# PREPARE INPUT DATA (EXACTLY 12 FEATURES IN CORRECT ORDER)
# ============================================================

input_data = pd.DataFrame([{
    "Air_Pollution": air_pollution,
    "Smoking": smoking,
    "Alcohol_Use": alcohol_use,
    "Diet_Salted_Processed": diet_salted_processed,
    "Obesity": obesity,
    "Diet_Red_Meat": diet_red_meat,
    "Age": age,
    "BMI": bmi,
    "Family_History": family_history_encoded,
    "Fruit_Veg_Intake": fruit_veg_intake,
    "Physical_Activity_Level": physical_inactivity,
    "BRCA_Mutation": brca_mutation_encoded
}])

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    # Load your ridge model (saved with joblib)
    model = joblib.load('model.pkl')  # Make sure this file exists
    return model

model = load_model()

# Make prediction
prediction = model.predict(input_data)[0]

# ============================================================
# DISPLAY RESULTS
# ============================================================

st.header("Prediction Results")

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Predicted Risk Score', f"{prediction:.3f}")
    
with col2:
    # Risk category based on score
    if prediction < 0.7:
        risk_category = "Low Risk 🟢"
    elif prediction < 1:
        risk_category = "Moderate Risk 🟡"
    else:
        risk_category = "High Risk 🔴"
    st.metric('Risk Category', risk_category)
    
with col3:
    st.metric('Scale', "0.0 - 1.0")

# ============================================================
# RISK FACTOR BREAKDOWN
# ============================================================

st.subheader("Your Risk Factors Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Lifestyle Factors**")
    st.write(f"🚬 Smoking: {smoking}/10")
    st.write(f"🍷 Alcohol: {alcohol_use}/10")
    st.write(f"🍔 Red Meat: {diet_red_meat}/10")
    st.write(f"🍟 Processed Foods: {diet_salted_processed}/10")
    st.write(f"🥬 Fruits/Veggies: {fruit_veg_intake}/10")
    st.write(f"🏃 Physical Inactivity: {physical_inactivity}/10")

with col2:
    st.markdown("**Health & Environmental Factors**")
    st.write(f"💨 Air Pollution: {air_pollution}/10")
    st.write(f"⚖️ Obesity Level: {obesity}/10")
    st.write(f"📊 BMI: {bmi:.1f}")
    st.write(f"🎂 Age: {age}")
    st.write(f"🧬 BRCA Mutation: {brca_mutation}")
    st.write(f"👨‍👩‍👧 Family History: {family_history}")

# ============================================================
# RECOMMENDATIONS BASED ON PREDICTION
# ============================================================

st.subheader("Recommendations")

if prediction < 0.7:
    st.success("✅ **Low Risk** - Maintain healthy habits!")
    st.write("• Continue regular exercise and balanced diet")
    st.write("• Annual check-ups recommended")
    st.write("• Keep up with cancer screenings as recommended by age")
    
elif prediction < 1:
    st.warning("⚠️ **Moderate Risk** - Consider lifestyle improvements")
    st.write("• Increase physical activity (aim for 150 min/week)")
    st.write("• Reduce processed meat and alcohol consumption")
    st.write("• Add more fruits and vegetables to your diet")
    st.write("• Consider genetic counseling if family history is concerning")
    
else:
    st.error("🔴 **High Risk** - Consult a healthcare provider")
    st.write("• Schedule a comprehensive cancer screening")
    st.write("• Discuss lifestyle modifications with your doctor")
    st.write("• Consider genetic testing for BRCA and other mutations")
    st.write("• Quit smoking and reduce alcohol intake immediately")
    st.write("• Maintain a healthy weight through diet and exercise")

# ============================================================
# FEATURE IMPACT EXPLANATION
# ============================================================

with st.expander("📊 How each factor affects your risk"):
    st.markdown("""
    **Factors that INCREASE cancer risk (higher score = higher risk):**
    - High air pollution exposure
    - Smoking
    - High alcohol consumption
    - Processed/salted foods
    - Obesity
    - Red meat consumption
    - Older age
    - Family history of cancer
    - BRCA gene mutation
    
    **Factors that DECREASE cancer risk (protective):**
    - High fruit & vegetable intake
    - Regular physical activity
    - Healthy BMI
    """)

# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")
st.caption("⚠️ **Disclaimer**: This prediction is based on statistical modeling and should not replace professional medical advice. Always consult with a healthcare provider for medical decisions.")