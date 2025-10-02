# VI-Data-Science-Home-Assignment

## Churn Prediction System - Recommendation Engine

A machine learning solution for identifying users with high churn probability based on behavioral features such as app visits, web visits, registration time, and more.

## Overview

This project implements an end-to-end machine learning pipeline for churn prediction, including:
- **Data Preparation**: Cleaning, normalization, and feature engineering
- **Model Training**: Multiple classical ML models implementations and hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and visualizations
- **High-Risk User Identification**: Recommendation system for targeting at-risk users 

## Features

### Data collecion Class

- **Feature Engineering**:
  - Normalization using StandardScaler
  - Label encoding for categorical features
  - Feature importance analysis using Random Forest
- **Data Validation**: Ensures data quality throughout the pipeline
  
### Data preprocessing Class
  - Min-Max Normalization 
  - Handle missing values 

### Model Training
- **Multiple Model Support**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier using lightGBM
- **Hyperparameter Tuning**: Grid search with cross-validation

### Model Evaluation
- **Performance Metrics**:
  - Accuracy, ROC-AUC Score, PR-AUC Score, logloss
- **Visualizations**:
  - ROC Curve
  - Precision-Recall Curve
  - Calibration (qq) plot

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/idotoren/VI-Data-Science-Home-Assignment.git
cd VI-Data-Science-Home-Assignment
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

Simply execute the main script:
```bash
python main.py
```

This will:
1. Collect data from csv files 
2. Perform feature engineering 
3. Train Logistic regression model with hyperparameter tuning
5. Evaluate model performance
6. Generate visualizations
7. Estimate optimal number of users for outreach
8. Identify and save high-risk users to csv file 

Should you want to run a different model, change the 'model_type' param in main.py

### Visualization Files are found in `Outputs/` directory

