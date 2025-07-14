# Customer-Churn-Prediction-Using-Machine-Learning


# 📊 Customer Churn Prediction - Machine Learning Project

This project aims to predict customer churn using machine learning models. We built and compared two models — `RandomForestClassifier` and `XGBoostClassifier` — to evaluate customer retention risk based on various telecom usage patterns and demographics.

---

## 📌 Problem Statement

Customer churn is a key concern in the telecom industry. The goal is to identify customers who are likely to leave (churn), allowing the company to take preventive action and improve customer retention.

---

## 🎯 Objectives

* Build a model to predict the likelihood of customer churn.
* Improve the recall of churners (class 1), especially in an imbalanced dataset.
* Compare basic vs advanced ML models for performance.
* Visualize and analyze model performance for business insights.

---

## 🧪 Hypothesis

* Customers with shorter contracts, higher bills, or poor service (e.g., no tech support) are more likely to churn.
* Providing better support and long-term contract offers can reduce churn.

---

## 🗃️ Dataset Overview

We created our own synthetic dataset based on real-world churn indicators, including:

| Feature         | Description                          |
| --------------- | ------------------------------------ |
| Gender          | Male / Female                        |
| SeniorCitizen   | 0 or 1                               |
| MonthlyCharges  | Monthly billing amount               |
| TotalCharges    | Lifetime billing amount              |
| InternetService | DSL / Fiber optic / No               |
| Contract        | Month-to-month / One year / Two year |
| TechSupport     | Yes / No                             |
| PaymentMethod   | Type of billing method               |
| ...and more     |                                      |

Target column: **`Churn`** (0 = Not Churned, 1 = Churned)

---

## ⚙️ Tools & Libraries

* Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
* Machine Learning Models: `RandomForestClassifier`, `XGBClassifier`
* Evaluation: Accuracy, Precision, Recall, F1-Score, ROC AUC
* Optional: `SMOTE` for class imbalance, Streamlit for UI

---

## 📈 Model Performance

| Metric              | RandomForest | XGBoost      |
| ------------------- | ------------ | ------------ |
| Accuracy            | 76.9%        | 76.0%        |
| Precision (Class 1) | 58%          | 54%          |
| Recall (Class 1)    | 46%          | 63% ✅        |
| ROC AUC Score       | \~0.79       | **0.8016 ✅** |

---

## 🔍 Key Findings

* **XGBoost** outperformed RandomForest in **recall for churners**, making it better for minimizing customer loss.
* **Contract type**, **TechSupport**, and **MonthlyCharges** were key predictors.
* Lowering prediction threshold from 0.5 to 0.4 improved the recall significantly.

---

## 💡 Recommendations

* Proactively engage customers with month-to-month plans and no tech support.
* Offer bundled discounts for Fiber customers with high charges.
* Prioritize outreach based on churn risk predicted by the model.

---

Author 
vishwanath kulal
SCEM,Mangalore

* A `requirements.txt` file
* Streamlit app setup steps
* GitHub folder structure template?

Let me know and I’ll generate them for you!
