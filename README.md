# 🛍️ Smart Retail Intelligence System (SRIS)

## 📌 Project Overview

Smart Retail Intelligence System (SRIS) is an End-to-End Machine Learning project designed to help retail businesses analyze customer behavior and make data-driven decisions.

The system combines multiple Machine Learning techniques into one interactive Streamlit application to:

- Segment customers
- Predict customer churn
- Forecast customer spending
- Visualize business insights

This project demonstrates practical implementation of Machine Learning models with an interactive user interface.

---

# 🚀 Features

## ✅ Customer Segmentation
- Uses KMeans Clustering
- Groups customers into:
  - Premium Customers
  - Regular Customers
  - Low Value Customers

## ✅ Churn Prediction
- Predicts whether a customer is likely to leave
- Uses Random Forest Classifier
- Displays churn probability

## ✅ Spending Prediction
- Forecasts estimated customer spending
- Uses Linear Regression

## ✅ Interactive Dashboard
- Built using Streamlit
- Real-time predictions
- Business-friendly visualizations

## ✅ Model Evaluation
Includes:
- Accuracy
- Precision
- Recall
- F1 Score
- MAE
- RMSE
- R² Score
- Silhouette Score

---

# 🧠 Machine Learning Techniques Used

| Technique | Algorithm | Purpose |
|---|---|---|
| Clustering | KMeans | Customer Segmentation |
| Classification | Random Forest | Churn Prediction |
| Regression | Linear Regression | Spending Prediction |

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

# 📂 Project Structure

```bash
Smart-Retail-Intelligence-System/
│
├── app.py
├── churn_modelling.csv
├── requirements.txt
├── README.md
└── models/
