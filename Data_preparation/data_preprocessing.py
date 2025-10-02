import numpy as np
import pandas as pd

class DataPreprocessing:
    def __init__(self, target_column='churn'):
        self.target_column = target_column

    def _handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        print("\nHandling missing values...")

        # Identify numeric and categorical columns
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features.remove('member_id')
        if self.target_column in self.numeric_features:
            self.numeric_features.remove(self.target_column)

        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()

        # For numeric features: fill with 0
        for col in self.numeric_features:
            if df[col].isnull().sum() > 0:
                # median_val = df[col].median()
                df[col].fillna(0, inplace=True)
                # print(f"  Filled {col} with median: {median_val}")

        # For categorical features: fill with 'other'
        for col in self.categorical_features:
            if df[col].isnull().sum() > 0:
                df[col].fillna('other', inplace=True)
                print(f"  Filled {col} with 'other'")

        return df

    def _normalize_data(self, df):
        print("\nNormalizing numerical features...")
        for col in self.numeric_features:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:  # Avoid division by zero
                df[col] = (df[col] - min_val) / (max_val - min_val)
                print(f"  Normalized {col}")
        return df


    def run(self, data):
        """
        Complete data preparation pipeline
        Returns: X (features) and y (target)
        """
        print("\n=== Starting Data Preprocessing Pipeline ===\n")

        # Step 1: Handle missing values
        df = self._handle_missing_values(data)

        # Step 2: Normalize numerical features
        df = self._normalize_data(df)

        print(f"\n=== Data Preprocessing Complete ===")

        return df
