# MLReg — Interactive Machine Learning Regression App

**MLReg** is an interactive machine learning regression platform that simplifies the complete regression workflow into a single, user-friendly environment. It allows users to upload datasets, preprocess data, train multiple regression models, compare performance metrics, and generate AI-assisted interpretations of results.

This project was developed as part of an innovation-focused machine learning workflow tool and is designed to support education, research, and rapid model experimentation.

---

# Overview

Regression modeling typically involves multiple steps, including data cleaning, preprocessing, model training, validation, and result interpretation. These steps often require programming expertise and the use of multiple tools.

**MLReg integrates these steps into one interactive application**, enabling users to perform regression analysis efficiently without needing to write extensive code.

The application is built using Python and deployed using Streamlit, allowing real-time interaction with data and machine learning models through a web-based interface.

---

# Key Features

## Data Handling

* Upload datasets in **CSV format**
* Automatic detection of **numeric columns**
* Missing value visualization
* Dataset summary statistics
* Correlation analysis tools

---

## Data Preprocessing

* Feature and target selection
* Missing value inspection
* Feature scaling support
* Train/Test dataset splitting

---

## Regression Models Supported

MLReg supports a wide range of regression algorithms across different modeling families:

### Linear Models

* Linear Regression
* Lasso Regression
* Ridge Regression
* Elastic Net Regression

### Instance-Based Methods

* K-Nearest Neighbors (KNN)

### Kernel-Based Models

* Support Vector Regression (SVR)

### Ensemble Models

* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression

### Neural Network Models

* Multi-Layer Perceptron (MLP) Regressor

This diverse model selection enables flexible experimentation and robust model comparison.

---

## Model Evaluation

Each trained model is evaluated using:

* R² (Coefficient of Determination)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

Users can store multiple trained models during a session and compare performance results interactively.

---

## AI-Powered Analysis

MLReg includes AI-assisted interpretation tools that automatically generate explanations for:

* Dataset statistics
* Correlation relationships
* Residual diagnostics
* Model performance comparison

These explanations help users better understand model behavior and identify common issues such as:

* Outliers
* Multicollinearity
* Overfitting
* Heteroscedasticity

---

# Application Workflow

The typical workflow in MLReg includes:

1. Upload CSV dataset
2. Explore dataset statistics
3. Select features and target variable
4. Apply preprocessing
5. Split dataset into train/test sets
6. Train regression models
7. Evaluate model performance
8. Generate AI-based explanations
9. Compare multiple trained models


---

# Intended Users

MLReg is designed for:

* Students learning machine learning regression
* Researchers performing data experiments
* Engineers analyzing predictive models
* Educators demonstrating regression workflows
* Data analysts exploring modeling strategies

---


# License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.


