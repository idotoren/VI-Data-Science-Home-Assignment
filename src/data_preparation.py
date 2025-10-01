"""
Data Preparation Module for Churn Prediction System
This module handles data cleaning, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class DataPreparation:
    """
    Handles all data preparation tasks including:
    - Data cleaning (missing values, duplicates)
    - Feature engineering (normalization, encoding)
    - Feature importance analysis
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_rows - df.shape[0]
        print(f"Removed {removed} duplicate rows")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        print("\nHandling missing values...")
        
        # Identify numeric and categorical columns
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'churn' in self.numeric_features:
            self.numeric_features.remove('churn')
        
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # For numeric features: fill with median
        for col in self.numeric_features:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} with median: {median_val}")
        
        # For categorical features: fill with mode
        for col in self.categorical_features:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  Filled {col} with mode: {mode_val}")
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features using Label Encoding"""
        print("\nEncoding categorical features...")
        
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Encoded {col}: {len(le.classes_)} unique values")
        
        return df
    
    def normalize_features(self, X_train, X_test=None):
        """Normalize numeric features using StandardScaler"""
        print("\nNormalizing features...")
        
        # Fit on training data
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_train_normalized = pd.DataFrame(
            X_train_normalized, 
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            # Transform test data
            X_test_normalized = self.scaler.transform(X_test)
            X_test_normalized = pd.DataFrame(
                X_test_normalized,
                columns=X_test.columns,
                index=X_test.index
            )
            print(f"  Normalized {X_train.shape[1]} features")
            return X_train_normalized, X_test_normalized
        
        print(f"  Normalized {X_train.shape[1]} features")
        return X_train_normalized
    
    def calculate_feature_importance(self, X, y, top_n=10):
        """Calculate feature importance using Random Forest"""
        print("\nCalculating feature importance...")
        
        # Train a quick Random Forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print(f"\nTop {top_n} most important features:")
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def prepare_data(self, df, target_column='churn'):
        """
        Complete data preparation pipeline
        Returns: X (features) and y (target)
        """
        print("\n=== Starting Data Preparation Pipeline ===\n")
        
        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove member_id if it exists (identifier, not a feature)
        if 'member_id' in X.columns:
            X = X.drop(columns=['member_id'])
            print("Removed member_id column (identifier)")
        
        # Step 4: Encode categorical features
        X = self.encode_categorical_features(X)
        
        print(f"\n=== Data Preparation Complete ===")
        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def get_feature_names(self):
        """Return list of feature names after preprocessing"""
        return self.numeric_features + self.categorical_features
