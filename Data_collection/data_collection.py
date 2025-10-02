import os
import pandas as pd
from datetime import datetime
import numpy as np

class DataCollector:
    def __init__(self, data_folder):
        """
        Initializes the DataCollector with the path to the data folder.

        Args:
            data_folder (str): Path to the folder containing all input CSV files.
        """
        self.data_folder = data_folder


    def _prepare_app_usgae_features(self, app_usage_file):
        """
        Prepares features from the app_usage.csv file.

        Args:
            app_usage_file (str): Path to the app_usage.csv file.

        Returns:
            pd.DataFrame: DataFrame with features for each unique member_id.
        """
        # Load the data
        data = pd.read_csv(app_usage_file)

        # Ensure timestamp is in datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Sort data by member_id and timestamp
        data = data.sort_values(by=['member_id', 'timestamp'])

        # Group by member_id
        grouped = data.groupby('member_id')

        # Calculate features
        self.app_usage_features = grouped.agg(
            num_visits=('timestamp', 'count'),  # Number of app visits
            hour_of_day=('timestamp', lambda x: x.dt.hour.mode()[0]),  # Most common hour of day
            day_of_week=('timestamp', lambda x: x.dt.dayofweek.mode()[0])  # Most common day of week
        ).reset_index()

        self.app_usage_features['hour_sin'] = np.sin(2 * np.pi * self.app_usage_features['hour_of_day'] / 24)
        self.app_usage_features['hour_cos'] = np.cos(2 * np.pi * self.app_usage_features['hour_of_day'] / 24)

        # Calculate max and min time between visits
        def time_diff_stats(group):
            time_diffs = group['timestamp'].diff().dt.total_seconds() / 3600  # Time differences in hours
            return pd.Series({
                'max_time_diff': time_diffs.max(),
                'min_time_diff': time_diffs.min()
            })

        time_diff_features = grouped.apply(time_diff_stats).reset_index()

        # Calculate time from current time to last visit
        current_time = datetime.now()
        last_visit_time = grouped['timestamp'].max().reset_index()
        last_visit_time['time_from_last_visit'] = (current_time - last_visit_time['timestamp']).dt.total_seconds() / 3600

        # Merge features
        self.app_usage_features = self.app_usage_features.merge(time_diff_features, on='member_id')
        self.app_usage_features = self.app_usage_features.merge(last_visit_time[['member_id', 'time_from_last_visit']], on='member_id')


    def _prepare_churn_label_features(self, churn_file):
        """
        Prepares churn labels from the churn.csv file.

        Args:
            churn_file (str): Path to the churn.csv file.

        Returns:
            pd.DataFrame: DataFrame with churn labels for each member_id.
        """
        # Load the data
        self.churn_labels = pd.read_csv(churn_file)

        # Ensure timestamp is in datetime format
        self.churn_labels['signup_date'] = pd.to_datetime(self.churn_labels['signup_date'])

        # Calculate time from current time to last visit
        current_time = datetime.now()
        self.churn_labels['days_from_signup'] = (current_time - self.churn_labels['signup_date']).dt.days


    def _prepare_web_visit_features(self, web_visits_file):
        """
        Prepares features from the web_visits.csv file.

        Args:
            web_visits_file (str): Path to the web_visits.csv file.

        Returns:
            pd.DataFrame: DataFrame with features for each unique member_id.
        """
        # Load the data
        data = pd.read_csv(web_visits_file)

        # Ensure timestamp is in datetime format
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Extract hour of day
        data['hour_of_day'] = data['timestamp'].dt.hour

        # Group by member_id
        grouped = data.groupby('member_id')

        # Calculate features
        self.web_visits_features = grouped.agg(
            num_unique_urls=('url', 'nunique'),  # Number of unique URLs visited
            num_unique_categories=('title', 'nunique')  # Number of unique categories visited
        ).reset_index()

        # Create visit pattern vector (counts of visits per category)
        visit_pattern = data.groupby(['member_id', 'title']).size().unstack(fill_value=0)
        visit_pattern.columns = [f'category_{col}_count' for col in visit_pattern.columns]
        self.web_visits_features = self.web_visits_features.merge(visit_pattern, on='member_id', how='left')

        # Calculate mode of time of day per category
        mode_time_of_day = data.groupby(['member_id', 'title'])['hour_of_day'].agg(lambda x: x.mode()[0]).unstack(fill_value=0)
        mode_time_of_day.columns = [f'category_{col}_mode_hour' for col in mode_time_of_day.columns]
        self.web_visits_features = self.web_visits_features.merge(mode_time_of_day, on='member_id', how='left')

        # Calculate time between repeated visits in the same category
        def time_diff_stats(group):
            time_diffs = group.sort_values('timestamp').groupby('title')['timestamp'].diff().dt.total_seconds() / 3600
            return time_diffs.mean()

        time_diff_features = grouped.apply(time_diff_stats).reset_index(name='avg_time_between_visits')
        self.web_visits_features = self.web_visits_features.merge(time_diff_features, on='member_id', how='left')


    def _prepare_claims_features(self, claims_file, collapse_icd=True, icd_keep_list=None):
        """
        Prepares features from the claims.csv file.

        Args:
            claims_file (str): Path to the claims.csv file.
            collapse_icd (bool): Whether to collapse ICD codes not in the keep list to 'other'.
            icd_keep_list (list): List of ICD codes to keep. Others will be collapsed to 'other'.

        Returns:
            pd.DataFrame: DataFrame with features for each unique member_id.
        """
        # Load the data
        data = pd.read_csv(claims_file)

        # Ensure diagnosis_date is in datetime format
        data['diagnosis_date'] = pd.to_datetime(data['diagnosis_date'])

        # Collapse ICD codes if required
        if collapse_icd:
            if icd_keep_list is None:
                icd_keep_list = ['Z71.3', 'I10', 'E11.9']
            data['icd_code'] = data['icd_code'].where(data['icd_code'].isin(icd_keep_list), 'other')

        # Calculate time from first diagnosis per ICD code
        data['time_from_first_diagnosis'] = data.groupby(['member_id', 'icd_code'])['diagnosis_date'].transform(
            lambda x: (x - x.min()).dt.days
        )

        # Pivot table to create features
        features = data.pivot_table(
            index='member_id',
            columns='icd_code',
            values='time_from_first_diagnosis',
            aggfunc='max'
        ).fillna(0).reset_index()

        # Rename columns for clarity
        features.columns = [f'time_from_first_{col}' if col != 'member_id' else col for col in features.columns]

        data['num_diff_diagnoses'] = data.groupby('member_id')['icd_code'].transform('nunique')
        self.claims_features = features.merge(data[['member_id', 'num_diff_diagnoses']].drop_duplicates(), on='member_id', how='left')


    def run(self, data_folder):
        """
        Executes all feature preparation functions, merges the results, validates uniqueness, and prints a summary.

        Args:
            data_folder (str): Path to the folder containing all input CSV files.

        Returns:
            pd.DataFrame: Final merged dataset with all features.
        """
        # File paths
        app_usage_file = os.path.join(data_folder, 'app_usage.csv')
        web_visits_file = os.path.join(data_folder, 'web_visits.csv')
        claims_file = os.path.join(data_folder, 'claims.csv')
        churn_file = os.path.join(data_folder, 'churn_labels.csv')

        # Prepare features
        print("Preparing app usage features...")
        self._prepare_app_usgae_features(app_usage_file)

        print("Preparing web visit features...")
        self._prepare_web_visit_features(web_visits_file)

        print("Preparing claims features...")
        self._prepare_claims_features(claims_file)

        print("Preparing churn label features...")
        self._prepare_churn_label_features(churn_file)

        # Merge all DataFrames on member_id
        print("Merging all features into one dataset...")
        final_dataset = self.app_usage_features.merge(self.web_visits_features, on='member_id', how='outer')
        final_dataset = final_dataset.merge(self.claims_features, on='member_id', how='outer')
        final_dataset = final_dataset.merge(self.churn_labels, on='member_id', how='outer')

        # Validate uniqueness of member_id
        if final_dataset['member_id'].duplicated().any():
            raise ValueError("Duplicate member_id rows found in the final dataset!")

        # Print dataset summary
        print("\nDataset Summary:")
        print(f"Number of rows: {final_dataset.shape[0]}")
        print(f"Number of columns: {final_dataset.shape[1]}")
        print("Preview of the dataset:")
        print(final_dataset.head())

        return final_dataset

# Example usage
data_folder = '../Data'  # Adjust the path to your data folder
data_collector = DataCollector(data_folder)
final_dataset = data_collector.run(data_folder)