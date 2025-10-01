"""
Generate sample dataset for churn prediction
This creates a realistic dataset with features like app_visits, web_visits, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(n_samples=1000, output_path='data/customer_data.csv'):
    """
    Generate sample customer data for churn prediction
    
    Features:
    - member_id: Unique identifier
    - app_visits: Number of app visits per month
    - web_visits: Number of web visits per month
    - days_since_registration: Days since user registered
    - total_purchases: Total number of purchases
    - avg_purchase_value: Average purchase value
    - customer_service_calls: Number of customer service interactions
    - account_age_days: Age of account in days
    - platform_preference: Preferred platform (iOS, Android, Web)
    - subscription_type: Type of subscription (Free, Basic, Premium)
    - churn: Target variable (0 = not churned, 1 = churned)
    """
    
    np.random.seed(42)
    
    print(f"Generating {n_samples} sample records...")
    
    # Generate member IDs
    member_ids = [f"MEM{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # Generate features with realistic distributions
    data = {
        'member_id': member_ids,
        'app_visits': np.random.poisson(lam=15, size=n_samples),
        'web_visits': np.random.poisson(lam=8, size=n_samples),
        'days_since_registration': np.random.randint(1, 1095, size=n_samples),  # Up to 3 years
        'total_purchases': np.random.poisson(lam=5, size=n_samples),
        'avg_purchase_value': np.random.gamma(shape=2, scale=25, size=n_samples),
        'customer_service_calls': np.random.poisson(lam=2, size=n_samples),
        'account_age_days': np.random.randint(30, 1095, size=n_samples),
        'platform_preference': np.random.choice(['iOS', 'Android', 'Web'], size=n_samples, p=[0.4, 0.4, 0.2]),
        'subscription_type': np.random.choice(['Free', 'Basic', 'Premium'], size=n_samples, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn based on features (realistic correlations)
    # Higher churn probability for:
    # - Low app/web visits
    # - Many customer service calls
    # - Low purchases
    # - Free subscription
    
    churn_probability = np.zeros(n_samples)
    
    # Base churn rate
    churn_probability += 0.15
    
    # Low engagement increases churn
    churn_probability += (df['app_visits'] < 5).astype(int) * 0.2
    churn_probability += (df['web_visits'] < 3).astype(int) * 0.15
    
    # High customer service calls increase churn
    churn_probability += (df['customer_service_calls'] > 4).astype(int) * 0.25
    
    # Low purchases increase churn
    churn_probability += (df['total_purchases'] < 2).astype(int) * 0.2
    
    # Subscription type affects churn
    churn_probability += (df['subscription_type'] == 'Free').astype(int) * 0.15
    churn_probability -= (df['subscription_type'] == 'Premium').astype(int) * 0.1
    
    # New accounts have higher churn
    churn_probability += (df['account_age_days'] < 90).astype(int) * 0.1
    
    # Clip probabilities to [0, 1]
    churn_probability = np.clip(churn_probability, 0, 1)
    
    # Generate churn labels
    df['churn'] = np.random.binomial(1, churn_probability)
    
    # Introduce some missing values (realistic scenario)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'avg_purchase_value'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'customer_service_calls'] = np.nan
    
    # Introduce some duplicates (to test duplicate removal)
    duplicate_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)
    
    print(f"\nDataset statistics:")
    print(f"  Total records: {len(df)}")
    print(f"  Churn rate: {df['churn'].mean():.2%}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    
    return df


if __name__ == "__main__":
    generate_sample_data()
