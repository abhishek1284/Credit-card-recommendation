# bank.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and columns
model = joblib.load('credit_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Credit Card Approval Predictor")

# Section 1: Single Prediction
st.header("🔹 Predict for a Single User")

income = st.number_input("Annual Income", min_value=10000)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
emp_length = st.slider("Employment Length (years)", 0, 10)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", min_value=60000)
int_rate = st.slider("Interest Rate (%)", 5.0, 40.0)
default_on_file = st.selectbox("Previous Default on File", ["Y", "N"])
cred_hist = st.slider("Credit History Length (years)", 0, 30)

percent_income = loan_amnt / income if income else 0

# Create user data dictionary
user_data = {
    'person_income': income,
    'person_emp_length': emp_length,
    'loan_amnt': loan_amnt,
    'loan_int_rate': int_rate,
    'loan_percent_income': percent_income,
    'cb_person_default_on_file_Y': 1 if default_on_file == 'Y' else 0,
    'cb_person_cred_hist_length': cred_hist,
}

# One-hot encode for user input
input_data = pd.DataFrame(columns=model_columns)
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

if st.button("Predict Approval for Single User"):
    prediction = model.predict(input_data)[0]
    st.success("Approved!" if prediction == 1 else "Rejected")

# Section 2: Batch Prediction
st.header("📂 Predict from CSV File")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.write("Raw Uploaded Data:", raw_data.head())

    # Preprocess uploaded data
    def preprocess(data):
        data['loan_percent_income'] = data['loan_amnt'] / data['person_income']
        data['cb_person_default_on_file_Y'] = data['cb_person_default_on_file'].map({'Y': 1, 'N': 0})
        data = data.drop(['cb_person_default_on_file'], axis=1)

        # One-hot encoding
        data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade'], drop_first=False)
        for col in model_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[model_columns]
        return data

    processed_data = preprocess(raw_data)
    predictions = model.predict(processed_data)
    raw_data['Approval_Status'] = ['Approved' if pred == 1 else 'Rejected' for pred in predictions]

    st.subheader("📋 Prediction Results")
    st.dataframe(raw_data)

    approved = raw_data[raw_data['Approval_Status'] == 'Approved']
    st.subheader("✅ List of People to Give Credit Cards")
    st.dataframe(approved)

    # Show count of approved credit cards
    st.info(f"✅ Total Approved Credit Cards: {len(approved)}")

    # Allow CSV download
    csv = approved.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Approved List", data=csv, file_name='approved_applicants.csv', mime='text/csv')
