
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Loan Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Loan EDA & Prediction")
st.write(
    "Explore bank loan data and predict loan approval "
    "using Machine Learning."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("bank_loans.csv")

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------

# Create a copy for machine learning
ml_df = df.copy()

# Encode categorical columns
le = LabelEncoder()

categorical_cols = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status"
]

for col in categorical_cols:
    ml_df[col] = le.fit_transform(ml_df[col])

# Fill missing values
for col in ml_df.columns:
    if ml_df[col].dtype == "object":
        ml_df[col] = ml_df[col].fillna(ml_df[col].mode()[0])
    else:
        ml_df[col] = ml_df[col].fillna(ml_df[col].median())

# Features and target
X = ml_df.drop("Loan_Status", axis=1)
y = ml_df["Loan_Status"]

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select a page:",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Loan Prediction"
    ]
)

# --------------------------------------------------
# Home
# --------------------------------------------------

if menu == "Home":

    st.header("Welcome to Bank Loan Prediction")

    st.write(
        "This application performs Exploratory Data Analysis "
        "and Machine Learning on bank loan data."
    )

    st.write("### Main Features")

    st.write("📊 Dataset Overview")
    st.write("📈 Exploratory Data Analysis")
    st.write("🏦 Loan Approval Prediction")
    st.write("🤖 Logistic Regression Machine Learning Model")

    st.info(
        "This application is created for educational "
        "and project demonstration purposes."
    )

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------

elif menu == "Dataset Overview":

    st.header("📋 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    st.subheader("Dataset")

    st.dataframe(df)

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum().to_frame("Missing Values"))

# --------------------------------------------------
# EDA
# --------------------------------------------------

elif menu == "EDA":

    st.header("📊 Exploratory Data Analysis")

    # Loan Status Distribution
    st.subheader("Loan Approval Distribution")

    fig1, ax1 = plt.subplots()

    sns.countplot(
        x="Loan_Status",
        data=df,
        ax=ax1
    )

    ax1.set_title("Loan Approval Distribution")

    st.pyplot(fig1)

    # Applicant Income Distribution
    st.subheader("Applicant Income Distribution")

    fig2, ax2 = plt.subplots()

    sns.histplot(
        df["ApplicantIncome"],
        bins=10,
        kde=True,
        ax=ax2
    )

    ax2.set_title("Applicant Income Distribution")

    st.pyplot(fig2)

    # Credit History vs Loan Status
    st.subheader("Loan Status vs Credit History")

    fig3, ax3 = plt.subplots()

    sns.countplot(
        x="Credit_History",
        hue="Loan_Status",
        data=df,
        ax=ax3
    )

    ax3.set_title("Loan Status vs Credit History")

    st.pyplot(fig3)

# --------------------------------------------------
# Loan Prediction
# --------------------------------------------------

elif menu == "Loan Prediction":

    st.header("🏦 Loan Approval Prediction")

    st.write("Enter applicant details:")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        married = st.selectbox(
            "Married",
            ["Yes", "No"]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        applicant_income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000
        )

        coapplicant_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=0.0
        )

    with col2:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=150.0
        )

        loan_amount_term = st.number_input(
            "Loan Amount Term",
            min_value=0.0,
            value=360.0
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"]
        )

    if st.button("Predict Loan Status"):

        # Convert input using the same LabelEncoder approach
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        # Encode categorical values
        for col in [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Property_Area"
        ]:
            input_data[col] = le.fit_transform(
                ml_df[col].astype(str)
            )

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("Prediction: Loan Approved ✅")
        else:
            st.error("Prediction: Loan Not Approved ❌")
