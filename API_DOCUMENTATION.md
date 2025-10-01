# API Documentation

## Churn Prediction System - Module Reference

This document provides detailed documentation for all modules and classes in the churn prediction system.

---

## Module: `data_preparation.py`

### Class: `DataPreparation`

Handles all data preparation tasks including data cleaning, preprocessing, and feature engineering.

#### Methods

##### `__init__()`
Initialize the DataPreparation class.

```python
data_prep = DataPreparation()
```

##### `load_data(filepath)`
Load data from a CSV file.

**Parameters:**
- `filepath` (str): Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataframe

**Example:**
```python
df = data_prep.load_data('data/customer_data.csv')
```

##### `remove_duplicates(df)`
Remove duplicate rows from dataframe.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with duplicates removed

##### `handle_missing_values(df)`
Handle missing values using appropriate strategies:
- Numeric features: Fill with median
- Categorical features: Fill with mode

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with missing values handled

##### `encode_categorical_features(df)`
Encode categorical features using Label Encoding.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with encoded categorical features

##### `normalize_features(X_train, X_test=None)`
Normalize numeric features using StandardScaler.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `X_test` (pd.DataFrame, optional): Test features

**Returns:**
- `pd.DataFrame` or `tuple`: Normalized features

**Example:**
```python
X_train_norm, X_test_norm = data_prep.normalize_features(X_train, X_test)
```

##### `calculate_feature_importance(X, y, top_n=10)`
Calculate feature importance using Random Forest.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target variable
- `top_n` (int): Number of top features to display

**Returns:**
- `pd.DataFrame`: Feature importance dataframe

##### `prepare_data(df, target_column='churn')`
Complete data preparation pipeline.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `target_column` (str): Name of target column

**Returns:**
- `tuple`: (X, y) - Features and target

**Example:**
```python
X, y = data_prep.prepare_data(df, target_column='churn')
```

---

## Module: `model_training.py`

### Class: `ModelTraining`

Handles model training, hyperparameter tuning, and model selection.

#### Methods

##### `__init__()`
Initialize the ModelTraining class with default models.

```python
trainer = ModelTraining()
```

##### `split_data(X, y, test_size=0.2, random_state=42)`
Split data into training and testing sets.

**Parameters:**
- `X` (pd.DataFrame): Features
- `y` (pd.Series): Target variable
- `test_size` (float): Proportion of test set (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Returns:**
- `tuple`: (X_train, X_test, y_train, y_test)

**Example:**
```python
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
```

##### `compare_models(X_train, y_train, cv=5)`
Compare different models using cross-validation.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `cv` (int): Number of cross-validation folds (default: 5)

**Returns:**
- `tuple`: (results_dict, best_model_name)

##### `tune_hyperparameters(X_train, y_train, model_name='Random Forest')`
Tune hyperparameters using GridSearchCV.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `model_name` (str): Name of model to tune

**Returns:**
- Trained model with best parameters

##### `train_model(X_train, y_train, model_name='Random Forest', tune=True)`
Train the final model.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series): Training target
- `model_name` (str): Name of model to train
- `tune` (bool): Whether to perform hyperparameter tuning

**Returns:**
- Trained model

**Example:**
```python
model = trainer.train_model(X_train, y_train, model_name='Random Forest', tune=True)
```

##### `predict(X)`
Make predictions using the trained model.

**Parameters:**
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.array`: Predicted labels

##### `predict_proba(X)`
Get prediction probabilities.

**Parameters:**
- `X` (pd.DataFrame): Features for prediction

**Returns:**
- `np.array`: Prediction probabilities

##### `save_model(filepath)`
Save the trained model to disk.

**Parameters:**
- `filepath` (str): Path to save the model

**Example:**
```python
trainer.save_model('models/churn_model.pkl')
```

##### `load_model(filepath)`
Load a trained model from disk.

**Parameters:**
- `filepath` (str): Path to the saved model

**Returns:**
- Loaded model

---

## Module: `model_evaluation.py`

### Class: `ModelEvaluation`

Handles model evaluation and performance visualization.

#### Methods

##### `__init__()`
Initialize the ModelEvaluation class.

```python
evaluator = ModelEvaluation()
```

##### `evaluate_model(y_true, y_pred, y_pred_proba=None)`
Comprehensive model evaluation.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred` (np.array): Predicted labels
- `y_pred_proba` (np.array, optional): Prediction probabilities

**Returns:**
- `dict`: Dictionary of evaluation metrics

**Example:**
```python
metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
```

##### `print_classification_report(y_true, y_pred)`
Print detailed classification report.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred` (np.array): Predicted labels

##### `plot_confusion_matrix(y_true, y_pred, save_path=None)`
Plot confusion matrix.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred` (np.array): Predicted labels
- `save_path` (str, optional): Path to save the plot

##### `plot_roc_curve(y_true, y_pred_proba, save_path=None)`
Plot ROC curve.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred_proba` (np.array): Prediction probabilities
- `save_path` (str, optional): Path to save the plot

##### `plot_feature_importance(feature_importance_df, top_n=15, save_path=None)`
Plot feature importance.

**Parameters:**
- `feature_importance_df` (pd.DataFrame): Feature importance dataframe
- `top_n` (int): Number of top features to plot
- `save_path` (str, optional): Path to save the plot

##### `plot_precision_recall_curve(y_true, y_pred_proba, save_path=None)`
Plot precision-recall curve.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred_proba` (np.array): Prediction probabilities
- `save_path` (str, optional): Path to save the plot

##### `generate_evaluation_report(y_true, y_pred, y_pred_proba, feature_importance_df=None, output_dir='./')`
Generate complete evaluation report with all visualizations.

**Parameters:**
- `y_true` (np.array): True labels
- `y_pred` (np.array): Predicted labels
- `y_pred_proba` (np.array): Prediction probabilities
- `feature_importance_df` (pd.DataFrame, optional): Feature importance dataframe
- `output_dir` (str): Directory to save outputs

**Returns:**
- `dict`: Dictionary of evaluation metrics

**Example:**
```python
metrics = evaluator.generate_evaluation_report(
    y_test, y_pred, y_pred_proba,
    feature_importance_df=feature_importance_df,
    output_dir='outputs'
)
```

---

## Module: `generate_data.py`

### Function: `generate_sample_data(n_samples=1000, output_path='data/customer_data.csv')`

Generate sample customer data for churn prediction.

**Parameters:**
- `n_samples` (int): Number of samples to generate (default: 1000)
- `output_path` (str): Path to save the generated data

**Returns:**
- `pd.DataFrame`: Generated dataframe

**Example:**
```python
from src.generate_data import generate_sample_data

df = generate_sample_data(n_samples=5000, output_path='data/large_dataset.csv')
```

---

## Complete Usage Example

```python
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation

# 1. Prepare data
data_prep = DataPreparation()
df = data_prep.load_data('data/customer_data.csv')
X, y = data_prep.prepare_data(df)

# 2. Train model
trainer = ModelTraining()
X_train, X_test, y_train, y_test = trainer.split_data(X, y)
X_train_norm, X_test_norm = data_prep.normalize_features(X_train, X_test)
model = trainer.train_model(X_train_norm, y_train)

# 3. Evaluate model
evaluator = ModelEvaluation()
y_pred = trainer.predict(X_test_norm)
y_pred_proba = trainer.predict_proba(X_test_norm)
metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)

# 4. Save model
trainer.save_model('models/my_model.pkl')
```

---

## Error Handling

All modules include appropriate error handling. Common exceptions:

- `ValueError`: Raised when invalid parameters are provided
- `FileNotFoundError`: Raised when trying to load non-existent files
- `KeyError`: Raised when expected columns are missing from dataframe

Always wrap calls in try-except blocks for production use:

```python
try:
    model = trainer.train_model(X_train, y_train)
except ValueError as e:
    print(f"Training error: {e}")
```
