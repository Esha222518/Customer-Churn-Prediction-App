import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📊 Customer Churn Prediction App")

def user_input():
    st.sidebar.header("📝 Enter Customer Info")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phoneservice = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiplelines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internetservice = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    onlinesecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    onlinebackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    deviceprotection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    techsupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streamingtv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streamingmovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    paymentmethod = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
    total = st.sidebar.slider("Total Charges", 0, 10000, 1000)

    data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phoneservice],
        'MultipleLines': [multiplelines],
        'InternetService': [internetservice],
        'OnlineSecurity': [onlinesecurity],
        'OnlineBackup': [onlinebackup],
        'DeviceProtection': [deviceprotection],
        'TechSupport': [techsupport],
        'StreamingTV': [streamingtv],
        'StreamingMovies': [streamingmovies],
        'Contract': [contract],
        'PaperlessBilling': [paperless],
        'PaymentMethod': [paymentmethod],
        'MonthlyCharges': [monthly],
        'TotalCharges': [total]
    })

    # Apply label encoders
    for col in data.columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    return data

# Get input from sidebar
input_df = user_input()

# Show encoded input for reference
st.subheader("📦 Encoded Input Features")
st.dataframe(input_df)

# Predict button
if st.button("🔮 Predict Churn"):
    try:
        churn_index = list(model.classes_).index(1)  # index of class "1" (churn)
        probability = model.predict_proba(input_df)[0][churn_index]

        st.subheader(f"🔎 Churn Probability: {round(probability * 100, 2)}%")
        st.text(f"Raw Model Output (predict_proba): {model.predict_proba(input_df)}")

        fig, ax = plt.subplots()
        ax.barh(["Churn", "No Churn"], [probability, 1 - probability], color=["red", "green"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
