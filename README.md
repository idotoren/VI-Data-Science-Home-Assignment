# VI-Data-Science-Home-Assignment

## Churn Prediction System - Recommendation Engine

A machine learning solution for identifying users with high churn probability based on behavioral features such as app visits, web visits, registration time, and more.

## Overview

This project implements a complete end-to-end machine learning pipeline for churn prediction, including:
- **Data Preparation**: Cleaning, normalization, encoding, and feature engineering
- **Model Training**: Multiple model comparison and hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and visualizations
- **High-Risk User Identification**: Recommendation system for targeting at-risk users

## Features

### Data Preparation Class
- **Data Cleaning**:
  - Remove duplicate records
  - Handle missing values (median for numeric, mode for categorical)
- **Feature Engineering**:
  - Normalization using StandardScaler
  - Label encoding for categorical features
  - Feature importance analysis using Random Forest
- **Data Validation**: Ensures data quality throughout the pipeline

### Model Training
- **Multiple Model Support**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Comparison**: Automatic selection of best performing model

### Model Evaluation
- **Performance Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Score
- **Visualizations**:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance Plot

## Project Structure

```
VI-Data-Science-Home-Assignment/
├── main.py                      # Main pipeline orchestration
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── __init__.py             # Package initialization
│   ├── data_preparation.py     # Data cleaning and preprocessing
│   ├── model_training.py       # Model training and tuning
│   ├── model_evaluation.py     # Model evaluation and visualization
│   └── generate_data.py        # Sample data generation
├── data/
│   └── customer_data.csv       # Generated sample dataset
├── models/
│   └── churn_model.pkl         # Trained model (generated)
└── outputs/
    ├── confusion_matrix.png    # Evaluation visualizations
    ├── roc_curve.png
    ├── precision_recall_curve.png
    └── feature_importance.png
```

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
1. Generate sample data (if not exists)
2. Perform data preparation and cleaning
3. Compare multiple models
4. Train the best model with hyperparameter tuning
5. Evaluate model performance
6. Generate visualizations
7. Identify high-risk users for recommendations

### Using Individual Components

#### Data Preparation
```python
from src.data_preparation import DataPreparation

data_prep = DataPreparation()
df = data_prep.load_data('data/customer_data.csv')
X, y = data_prep.prepare_data(df, target_column='churn')
```

#### Model Training
```python
from src.model_training import ModelTraining

trainer = ModelTraining()
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
model = trainer.train_model(X_train, y_train, model_name='Random Forest')
```

#### Model Evaluation
```python
from src.model_evaluation import ModelEvaluation

evaluator = ModelEvaluation()
y_pred = trainer.predict(X_test)
y_pred_proba = trainer.predict_proba(X_test)
metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
```

## Dataset Features

The system uses the following features for churn prediction:

| Feature | Type | Description |
|---------|------|-------------|
| member_id | String | Unique user identifier |
| app_visits | Numeric | Number of mobile app visits per month |
| web_visits | Numeric | Number of web visits per month |
| days_since_registration | Numeric | Days since user registration |
| total_purchases | Numeric | Total number of purchases made |
| avg_purchase_value | Numeric | Average value of purchases |
| customer_service_calls | Numeric | Number of customer service interactions |
| account_age_days | Numeric | Age of account in days |
| platform_preference | Categorical | Preferred platform (iOS/Android/Web) |
| subscription_type | Categorical | Type of subscription (Free/Basic/Premium) |
| **churn** | Binary | **Target variable** (0 = not churned, 1 = churned) |

## Model Performance

The system typically achieves the following performance metrics (on sample data):
- **Accuracy**: ~85-90%
- **ROC-AUC**: ~0.85-0.92
- **Precision**: ~0.80-0.88
- **Recall**: ~0.75-0.85

## Output Files

After running the pipeline, the following files are generated:

### Model Files
- `models/churn_model.pkl`: Trained model ready for deployment

### Visualization Files (in `outputs/` directory)
- `confusion_matrix.png`: Model confusion matrix
- `roc_curve.png`: ROC curve with AUC score
- `precision_recall_curve.png`: Precision-recall trade-off
- `feature_importance.png`: Top important features for prediction

## Recommendations for Production

1. **Data Quality**: Ensure regular data validation and monitoring
2. **Model Retraining**: Retrain model periodically with new data
3. **Threshold Tuning**: Adjust churn probability threshold based on business needs
4. **A/B Testing**: Test retention campaigns on high-risk users
5. **Feature Engineering**: Add more behavioral and demographic features
6. **Ensemble Methods**: Consider stacking multiple models for better performance

## Dependencies

See `requirements.txt` for full list:
- pandas>=1.5.0
- numpy>=1.23.0
- scikit-learn>=1.2.0
- matplotlib>=3.6.0
- seaborn>=0.12.0
- joblib>=1.2.0

## License

This project is created as part of a data science home assignment.

## Author

VI Data Science Team