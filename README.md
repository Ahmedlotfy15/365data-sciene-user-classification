# ML Data Analysis and Classification Project

This project performs a comprehensive data analysis and applies various machine learning models to classify users based on their behaviors and features. Below is an overview of the project, steps taken, and key results.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Machine Learning Models](#machine-learning-models)

---

## Introduction

The goal of this project is to:
1. Perform exploratory data analysis to uncover patterns and trends.
2. Preprocess the data, handling outliers, missing values, and multicollinearity.
3. Train and evaluate machine learning models for user classification.
4. Identify the best-performing model using metrics like accuracy and classification reports.

---

## Dataset

The dataset contains user behavior data, such as:
- `minutes_watched`
- `practice_exams_started`
- `practice_exams_passed`
- `days_on_platform`
- `minutes_spent_on_exams`
- `student_country`
- `purchased` (target variable)

Source: `/kaggle/input/365-datascience-users-dataset/ml_datasource.csv`

---

## Data Preprocessing

### Steps:
1. **Outliers Handling**:
   - Removed outliers from numerical columns using visualizations (histograms, density plots, and box plots).
2. **Multicollinearity**:
   - Calculated Variance Inflation Factors (VIF) and dropped features with high VIF.
3. **Missing Values**:
   - Filled missing values in the `student_country` column with "NAM".
4. **Encoding**:
   - Applied `OrdinalEncoder` to encode categorical columns.
5. **Data Splitting**:
   - Split the data into training (80%) and testing (20%) sets.

---

## Exploratory Data Analysis (EDA)

Visualizations were created to better understand the data:
- Histograms and density plots for distributions.
- Box plots to detect outliers.
- Correlation matrix to study relationships between numerical features.

---

## Machine Learning Models

### 1. Logistic Regression
- Built using the `statsmodels` library.
- Model coefficients were analyzed, and performance was evaluated using confusion matrices and classification reports.

### 2. K-Nearest Neighbors (KNN)
- Tuned hyperparameters using GridSearchCV (`n_neighbors` and `weights`).
- Evaluated using confusion matrices and classification reports.

### 3. Support Vector Classifier (SVC)
- Applied MinMaxScaler for feature scaling.
- Tuned hyperparameters using GridSearchCV (`kernel`, `C`, `gamma`).

### 4. Decision Tree Classifier
- Optimized using `ccp_alpha` parameter and visualized the decision tree.

### 5. Random Forest Classifier
- Integrated `ccp_alpha` from the best Decision Tree model.
- Evaluated the ensemble model on the test set.

---
