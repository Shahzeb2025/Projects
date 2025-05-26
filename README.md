# 🚲 Bike Demand Prediction using Machine Learning & Deep Learning

This project focuses on predicting daily bike rental demand using a combination of machine learning and deep learning techniques. It includes comprehensive data preprocessing, feature engineering, and the implementation of both a Neural Network and a Random Forest Regressor, culminating in an ensemble model for improved prediction accuracy.

---

## 📌 Overview

The goal of this project is to accurately forecast the number of bikes rented daily based on various environmental and temporal features. By leveraging both traditional machine learning and modern deep learning approaches, the project aims to demonstrate robust predictive capabilities.

---

## 📊 Dataset Information

This dataset is sourced from the **UCI Machine Learning Repository**.

> Bike-sharing systems are modern alternatives to traditional bike rentals where processes such as membership, rental, and returns are automated. With over 500 programs globally, these systems are not only transforming transportation but also serving as virtual sensor networks that capture mobility patterns in cities.

These datasets are valuable for urban research, as they provide detailed logs including travel duration, departure, and arrival positions—allowing for meaningful data-driven insights.

---

## 🛠 Features and Methodology

### 📥 Data Loading and Preprocessing
- Load CSV file and convert date columns
- Extract relevant time-based features

### 🧠 Feature Engineering
- `is_weekend`: Identifies weekends
- `week_of_year`: Captures weekly seasonality
- `cnt_lag1`: Previous day's bike count
- `cnt_rolling_3`: 3-day rolling average bike count

### 🧹 Data Cleaning
- Removed irrelevant columns: `instant`, `dteday`, `yr`, `casual`, `registered`

### 🔄 Feature Transformation
- `MinMaxScaler`: Scales numerical features
- `OneHotEncoder`: Encodes categorical variables
- `ColumnTransformer`: Combines both pipelines

### 🏗️ Model Development

#### 🔹 Neural Network (Deep Learning)
- Built using **Keras Sequential API**
- Includes multiple dense layers with `Dropout` and `L2 Regularization`
- Trained using **Adam optimizer** and **Huber loss**
- **EarlyStopping** added to prevent overfitting

#### 🔹 Random Forest Regressor (Machine Learning)
- Built using **Scikit-learn**
- Ensemble learning technique, strong baseline model

#### 🔁 Ensemble Model
- Final prediction is the average of NN and RF model outputs

---

## 🧪 Model Evaluation

| Metric               | Neural Net | Random Forest | Ensemble |
|----------------------|------------|----------------|----------|
| MAE (Test)           | ~692.65    | ~179.79        | ~283.47  |
| R² Score (Test)      | -          | -              | **~0.957** |

✅ The ensemble model provided a well-balanced and highly accurate prediction.

---

## 💻 Technologies Used

- **Python**  
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Machine Learning**: `scikit-learn` (for preprocessing, training, evaluation)
- **Deep Learning**: `Keras`, `TensorFlow`

---

## 📂 Files in the Project

- `bike_demand_app.py` – Streamlit interface (saved but not deployed)
- `bike_model_final.h5` – Saved deep learning model
- `bike_rf_model.pkl` – Saved Random Forest model
- `bike_preprocessor.pkl` – Saved preprocessing pipeline
- `requirements.txt` – Project dependencies
- `README.md` – Project documentation

---

## 🔖 Project Summary (One-liner)

> I developed a bike rental demand prediction system using machine learning and deep learning models, and saved it as a Streamlit app for future local deployment.

---

## 🔮 Future Work

- Deploy the Streamlit app publicly via Streamlit Cloud
- Add model interpretability (e.g., SHAP values)
- Extend to hourly demand prediction
