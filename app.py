# ============================================
# SMART RETAIL INTELLIGENCE SYSTEM (SRIS)
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Smart Retail Intelligence System",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Smart Retail Intelligence System (SRIS)")
st.markdown("### End-to-End Machine Learning Project")

# ============================================
# LOAD DATA
# ============================================

DATA_PATH = "Churn_Modelling.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Dataset Error: {e}")
    st.stop()

# ============================================
# DATA CLEANING
# ============================================

# Remove unwanted columns
remove_cols = [
    'RowNumber',
    'CustomerId',
    'Surname',
    'apst'
]

for col in remove_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# ============================================
# CHECK REQUIRED COLUMNS
# ============================================

required_columns = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
    'Exited'
]

missing_cols = [
    col for col in required_columns
    if col not in df.columns
]

if missing_cols:
    st.error(f"❌ Missing Columns: {missing_cols}")
    st.stop()

# ============================================
# CLEAN TEXT DATA
# ============================================

df['Geography'] = (
    df['Geography']
    .astype(str)
    .str.strip()
)

df['Gender'] = (
    df['Gender']
    .astype(str)
    .str.strip()
)

# ============================================
# MANUAL ENCODING
# ============================================

geo_mapping = {
    "France": 0,
    "Germany": 1,
    "Spain": 2
}

gender_mapping = {
    "Female": 0,
    "Male": 1
}

df['Geography'] = df['Geography'].map(geo_mapping)
df['Gender'] = df['Gender'].map(gender_mapping)

# Fill missing values safely
df['Geography'] = df['Geography'].fillna(0)
df['Gender'] = df['Gender'].fillna(0)

# Convert datatype
df['Geography'] = df['Geography'].astype(int)
df['Gender'] = df['Gender'].astype(int)

# ============================================
# REMOVE NULL VALUES
# ============================================

df.dropna(inplace=True)

# ============================================
# FEATURE SELECTION
# ============================================

X = df.drop('Exited', axis=1)
y = df['Exited']

# ============================================
# EMPTY DATA CHECK
# ============================================

if X.shape[0] == 0:
    st.error("❌ Dataset became empty after preprocessing.")
    st.stop()

# ============================================
# CREATE SPENDING TARGET
# ============================================

np.random.seed(42)

df['EstimatedSpending'] = (
    df['Balance'] * 0.4 +
    df['EstimatedSalary'] * 0.5 +
    df['Age'] * 100
)

spending_target = df['EstimatedSpending']

# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
os.makedirs("models", exist_ok=True)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# ============================================
# CUSTOMER SEGMENTATION
# ============================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

silhouette = silhouette_score(
    X_scaled,
    clusters
)

# Save model
joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

cluster_names = {
    0: "Premium Customer",
    1: "Regular Customer",
    2: "Low Value Customer"
}

df['CustomerSegment'] = [
    cluster_names[c]
    for c in clusters
]

# ============================================
# CHURN PREDICTION MODEL
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

churn_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

churn_model.fit(X_train, y_train)

# Predictions
churn_pred = churn_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, churn_pred)

precision = precision_score(
    y_test,
    churn_pred
)

recall = recall_score(
    y_test,
    churn_pred
)

f1 = f1_score(
    y_test,
    churn_pred
)

# Save model
joblib.dump(
    churn_model,
    "models/churn_model.pkl"
)

# ============================================
# SPENDING PREDICTION MODEL
# ============================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled,
    spending_target,
    test_size=0.2,
    random_state=42
)

spending_model = LinearRegression()

spending_model.fit(
    X_train_reg,
    y_train_reg
)

# Predictions
spending_pred = spending_model.predict(
    X_test_reg
)

# Metrics
mae = mean_absolute_error(
    y_test_reg,
    spending_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        spending_pred
    )
)

r2 = r2_score(
    y_test_reg,
    spending_pred
)

# Save model
joblib.dump(
    spending_model,
    "models/spending_model.pkl"
)

# ============================================
# SIDEBAR
# ============================================

st.sidebar.header("📌 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Customer Segmentation",
        "Churn Prediction",
        "Spending Prediction",
        "Model Performance"
    ]
)

# ============================================
# DASHBOARD PAGE
# ============================================

if page == "Dashboard":

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Average Balance",
        f"${df['Balance'].mean():,.0f}"
    )

    col3.metric(
        "Churn Rate",
        f"{df['Exited'].mean()*100:.2f}%"
    )

    st.dataframe(df.head())

    st.subheader("📈 Customer Segments")

    fig, ax = plt.subplots(figsize=(8, 5))

    df['CustomerSegment'].value_counts().plot(
        kind='bar',
        ax=ax
    )

    ax.set_ylabel("Customers")

    st.pyplot(fig)

# ============================================
# CUSTOMER SEGMENTATION PAGE
# ============================================

elif page == "Customer Segmentation":

    st.subheader("👥 Customer Segmentation")

    st.write(
        "Silhouette Score:",
        round(silhouette, 3)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=df['Age'],
        y=df['Balance'],
        hue=df['CustomerSegment'],
        ax=ax
    )

    st.pyplot(fig)

    st.dataframe(
        df[
            [
                'Age',
                'Balance',
                'CustomerSegment'
            ]
        ].head(20)
    )

# ============================================
# CHURN PREDICTION PAGE
# ============================================

elif page == "Churn Prediction":

    st.subheader("⚠️ Customer Churn Prediction")

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    age = st.slider(
        "Age",
        18,
        80,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

    balance = st.number_input(
        "Balance",
        0.0,
        300000.0,
        50000.0
    )

    products = st.slider(
        "Number of Products",
        1,
        4,
        2
    )

    salary = st.number_input(
        "Estimated Salary",
        0.0,
        300000.0,
        60000.0
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    # Encode input
    geo_encoded = geo_mapping[geography]
    gender_encoded = gender_mapping[gender]

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geo_encoded],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [salary]
    })

    # Match training columns safely
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    if st.button("Predict Churn"):

        prediction = churn_model.predict(
            input_scaled
        )[0]

        probability = churn_model.predict_proba(
            input_scaled
        )[0][1]

        if prediction == 1:

            st.error(
                f"⚠️ High Churn Risk "
                f"({probability*100:.2f}%)"
            )

        else:

            st.success(
                f"✅ Low Churn Risk "
                f"({(1-probability)*100:.2f}%)"
            )

# ============================================
# SPENDING PREDICTION PAGE
# ============================================

elif page == "Spending Prediction":

    st.subheader("💰 Spending Prediction")

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )

    age = st.slider(
        "Age",
        18,
        80,
        40
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        6
    )

    balance = st.number_input(
        "Balance",
        0.0,
        300000.0,
        80000.0
    )

    products = st.slider(
        "Products",
        1,
        4,
        2
    )

    salary = st.number_input(
        "Salary",
        0.0,
        300000.0,
        70000.0
    )

    geography = st.selectbox(
        "Select Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Select Gender",
        ["Male", "Female"]
    )

    # Encode input
    geo_encoded = geo_mapping[geography]
    gender_encoded = gender_mapping[gender]

    spending_input = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geo_encoded],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [salary]
    })

    # Match columns safely
    for col in X.columns:
        if col not in spending_input.columns:
            spending_input[col] = 0

    spending_input = spending_input[X.columns]

    # Scale input
    spending_scaled = scaler.transform(
        spending_input
    )

    if st.button("Predict Spending"):

        spend_prediction = spending_model.predict(
            spending_scaled
        )[0]

        st.success(
            f"💵 Estimated Spending: "
            f"${spend_prediction:,.2f}"
        )

# ============================================
# MODEL PERFORMANCE PAGE
# ============================================

elif page == "Model Performance":

    st.subheader("📊 Model Performance")

    st.markdown("## 🔹 Churn Model")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        round(accuracy, 3)
    )

    col2.metric(
        "Precision",
        round(precision, 3)
    )

    col1.metric(
        "Recall",
        round(recall, 3)
    )

    col2.metric(
        "F1 Score",
        round(f1, 3)
    )

    st.markdown("---")

    st.markdown("## 🔹 Spending Model")

    col3, col4 = st.columns(2)

    col3.metric(
        "MAE",
        round(mae, 2)
    )

    col4.metric(
        "RMSE",
        round(rmse, 2)
    )

    st.metric(
        "R² Score",
        round(r2, 3)
    )

    st.markdown("---")

    st.markdown("## 🔹 Clustering")

    st.metric(
        "Silhouette Score",
        round(silhouette, 3)
    )

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    "Made with ❤️ using Python, Streamlit & Machine Learning"
)










