import numpy as np
import pandas as pd

class DataPreprocessing:
    def __init__(self, raw_data, target_column='churn'):
        self.data = raw_data
        self.target_column = target_column

    def _handle_missing_values(self, df):
        """Handle missing values using appropriate strategies"""
        print("\nHandling missing values...")

        # Identify numeric and categorical columns
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
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

    def _clean_data(self):
        # Implement data cleaning logic here
        pass

    def _normalize_data(self):
        # Implement data normalization logic here
        pass

    def run(self):
        """
        Complete data preparation pipeline
        Returns: X (features) and y (target)
        """
        print("\n=== Starting Data Preprocessing Pipeline ===\n")

        # Step 1: Handle missing values
        df = self._handle_missing_values(df)

        # Step 2: remove outliers
        df = self._remove_outliers(df)

        # Step 3: Separate features and target
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])

        # Remove member_id if it exists (identifier, not a feature)
        if 'member_id' in X.columns:
            X = X.drop(columns=['member_id'])
            print("Removed member_id column (identifier)")

        # # Step 4: Encode categorical features
        # X = self.encode_categorical_features(X)

        print(f"\n=== Data Preprocessing Complete ===")
        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")

        return X, y


    def _remove_outliers(self, df):
        return df
