import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set up the Streamlit page with a dark theme
st.set_page_config(
    page_title="Loan Status Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_pickle_file(filename):
    """Load a pickle file."""
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load the model and scaler
model = load_pickle_file("logistic_regression_model.pkl")
scaler = load_pickle_file("scaler.pkl")

# Feature input columns
input_cols = ['Gender', 'Married', 'Dependents', 'Education',
              'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
              'Loan_Amount_Term', 'Credit_History', 'Rural', 'Semiurban', 'Urban']

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Main App
st.markdown("# 💼 Loan Status Prediction App")
st.markdown(
    "### An interactive tool to predict loan approval using logistic regression."
)

# Sidebar
st.sidebar.header("User Inputs")
st.sidebar.markdown("### Provide details below:")

def user_input_features():
    Gender = st.sidebar.radio("Gender:", ("Male", "Female"))
    Married = st.sidebar.radio("Married:", ("Yes", "No"))
    Dependents = st.sidebar.selectbox("Dependents:", ("0", "1", "2", "3+"))
    Education = st.sidebar.radio("Education:", ("Graduate", "Not Graduate"))
    Self_Employed = st.sidebar.radio("Self-Employed:", ("Yes", "No"))
    ApplicantIncome = st.sidebar.number_input("Applicant Income:", min_value=0, step=500)
    CoapplicantIncome = st.sidebar.number_input("Coapplicant Income:", min_value=0, step=500)
    LoanAmount = st.sidebar.number_input("Loan Amount:", min_value=0, step=10)
    Loan_Amount_Term = st.sidebar.selectbox("Loan Amount Term:", (12, 36, 60, 120, 180, 240, 300, 360, 480))
    Credit_History = st.sidebar.radio("Credit History:", ("Good", "Bad"))
    Property_Area = st.sidebar.radio("Property Area:", ("Rural", "Semiurban", "Urban"))

    # Convert to DataFrame
    data = {
        'Gender': [1 if Gender == "Male" else 0],
        'Married': [1 if Married == "Yes" else 0],
        'Dependents': [0 if Dependents == "0" else (1 if Dependents == "1" else (2 if Dependents == "2" else 3))],
        'Education': [1 if Education == "Graduate" else 0],
        'Self_Employed': [1 if Self_Employed == "Yes" else 0],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [1 if Credit_History == "Good" else 0],
        'Rural': [1 if Property_Area == "Rural" else 0],
        'Semiurban': [1 if Property_Area == "Semiurban" else 0],
        'Urban': [1 if Property_Area == "Urban" else 0]
    }

    return pd.DataFrame(data)

# Collect input from the user
input_data = user_input_features()
st.markdown("## User Input Data")
st.write(input_data)

# Preprocess and scale input data
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data[input_cols])
    result = "Approved" if prediction[0] == 1 else "Rejected"

    # Display result
    st.markdown(f"## Prediction: Loan Status is **{result}**")
    
    if result == "Approved":
        st.success("Congratulations! Your loan has been approved.")
    else:
        st.error("Unfortunately, your loan application was rejected.")

# Footer
st.markdown("---")
st.markdown("### Created by:\n1) Mohammad Yesaullah Sheikh\n2) Emaan Arshad\n3) Tanisha Kataria")
