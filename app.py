
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("Best_Customer_Churn_Model.pkl")

# Application title
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")

st.write(
    "Predict whether a telecom customer is likely to leave the service."
)

st.sidebar.header("Customer Information")

# Customer inputs

gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen", ["Yes", "No"]
)

partner = st.sidebar.selectbox(
    "Partner", ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents", ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure Months", 0, 100, 12
)

phone = st.sidebar.selectbox(
    "Phone Service", ["Yes", "No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.number_input(
    "Monthly Charges", 0.0, 200.0, 70.0
)

total = st.sidebar.number_input(
    "Total Charges", 0.0, 10000.0, 840.0
)

# Prepare input data

input_data = pd.DataFrame([{

    "Gender": gender,
    "Senior Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone,
    "Multiple Lines": multiple,
    "Internet Service": internet,
    "Online Security": security,
    "Online Backup": backup,
    "Device Protection": device,
    "Tech Support": tech,
    "Streaming TV": tv,
    "Streaming Movies": movies,
    "Contract": contract,
    "Paperless Billing": paperless,
    "Payment Method": payment,
    "Monthly Charges": monthly,
    "Total Charges": total

}])

# Customer Churn Prediction

st.subheader("Customer Churn Prediction")

if st.button("Predict Customer Churn"):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error("Customer is likely to CHURN")
            else:
                st.success("Customer is likely to STAY")

        with col2:

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        st.subheader("Customer Input Details")

        st.dataframe(input_data)

    except Exception as e:

        st.error("Prediction failed!")

        st.exception(e)
