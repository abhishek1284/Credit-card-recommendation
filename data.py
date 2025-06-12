# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('credit_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Credit Card Approval Predictor")

# Collect user input
income = st.number_input("Annual Income", min_value=1000)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp_length = st.slider("Employment Length (years)", 0, 30)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=500)
int_rate = st.slider("Interest Rate (%)", 5.0, 40.0)
percent_income = loan_amnt / income
default_on_file = st.selectbox("Previous Default on File", ["Y", "N"])
cred_hist = st.slider("Credit History Length (years)", 0, 30)

# Build input DataFrame
input_data = pd.DataFrame(columns=model_columns)
user_data = {
    'person_income': income,
    'person_emp_length': emp_length,
    'loan_amnt': loan_amnt,
    'loan_int_rate': int_rate,
    'loan_percent_income': percent_income,
    'cb_person_default_on_file_Y': 1 if default_on_file == 'Y' else 0,
    'cb_person_cred_hist_length': cred_hist
}

# One-hot encode manually
for col in model_columns:
    if col in user_data:
        input_data.at[0, col] = user_data[col]
    elif 'person_home_ownership_' in col and home_ownership in col:
        input_data.at[0, col] = 1
    elif 'loan_intent_' in col and loan_intent in col:
        input_data.at[0, col] = 1
    elif 'loan_grade_' in col and loan_grade in col:
        input_data.at[0, col] = 1
    else:
        input_data.at[0, col] = 0

# Predict
if st.button("Predict Approval"):
    prediction = model.predict(input_data)[0]
    st.success("Approved!" if prediction == 1 else "Rejected")
