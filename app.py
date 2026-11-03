import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# STEP 1: PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.write(
    "Predict customer churn and evaluate five machine-learning models."
)

# --------------------------------------------------
# STEP 2: LOAD PREPROCESSOR AND TRAINED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    preprocessor = joblib.load("preprocessor.pkl")

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "SVM": "svm.pkl"
    }

    trained_models = {}

    for name, filename in model_files.items():
        path = os.path.join("model", filename)
        trained_models[name] = joblib.load(path)

    return preprocessor, trained_models


try:
    preprocessor, trained_models = load_models()
    st.success("All five trained models loaded successfully!")

except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()


# --------------------------------------------------
# STEP 3: MODEL EVALUATION SECTION
# --------------------------------------------------

st.header("1. Model Evaluation")

st.write(
    "Upload a test dataset containing 19 customer features "
    "and the actual Churn Value column."
)

uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Test Dataset")
        st.write("Dataset shape:", test_data.shape)
        st.dataframe(test_data.head())

        if "Churn Value" not in test_data.columns:
            st.error("The CSV must contain a 'Churn Value' column.")

        else:

            # Separate input features and actual target
            X_test = test_data.drop(columns=["Churn Value"])
            y_test = test_data["Churn Value"]

            # Convert target to numeric
            y_test = pd.to_numeric(y_test, errors="coerce")

            if y_test.isna().any():
                st.error("Churn Value must contain only numeric 0 and 1.")
                st.stop()

            if not set(y_test.unique()).issubset({0, 1}):
                st.error("Churn Value must contain only 0 and 1.")
                st.stop()

            # Apply the saved preprocessing
            X_test_processed = preprocessor.transform(X_test)

            # Select model
            selected_model_name = st.selectbox(
                "Select Machine Learning Model",
                list(trained_models.keys())
            )

            selected_model = trained_models[selected_model_name]

            # Generate predictions
            y_pred = selected_model.predict(X_test_processed)

            # Generate probabilities for AUC
            y_probability = selected_model.predict_proba(
                X_test_processed
            )[:, list(selected_model.classes_).index(1)]

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(
                y_test, y_pred, zero_division=0
            )

            recall = recall_score(
                y_test, y_pred, zero_division=0
            )

            f1 = f1_score(
                y_test, y_pred, zero_division=0
            )

            mcc = matthews_corrcoef(y_test, y_pred)

            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_probability)
            else:
                auc = float("nan")

            # Display results
            st.subheader("Model Evaluation Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1]
            )

            cm_df = pd.DataFrame(
                cm,
                index=["Actual No Churn", "Actual Churn"],
                columns=["Predicted No Churn", "Predicted Churn"]
            )

            st.dataframe(cm_df)

            # Classification report
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True
            )

            st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error during model evaluation: {e}")


# --------------------------------------------------
# STEP 4: INDIVIDUAL CUSTOMER PREDICTION
# --------------------------------------------------

st.divider()

st.header("2. Individual Customer Churn Prediction")

st.write(
    "Enter customer information to predict whether the customer "
    "is likely to leave the telecom service."
)

with st.form("customer_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )

        senior = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )

        partner = st.selectbox(
            "Partner", ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents", ["No", "Yes"]
        )

        tenure = st.number_input(
            "Tenure Months", min_value=0, max_value=72, value=12
        )

        phone = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn"
    )


if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total
    }])

    try:

        processed_input = preprocessor.transform(input_data)

        # Use the Logistic Regression model for individual prediction
        individual_model = trained_models["Logistic Regression"]

        prediction = individual_model.predict(processed_input)[0]

        probability = individual_model.predict_proba(
            processed_input
        )[0][list(individual_model.classes_).index(1)]

        st.subheader("Prediction Results")

        st.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        if prediction == 1:
            st.error("This customer is likely to CHURN.")
        else:
            st.success("This customer is likely to STAY.")

        st.subheader("Customer Input Details")
        st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")