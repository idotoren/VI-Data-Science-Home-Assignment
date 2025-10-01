"""
Example: Using the trained model to predict churn for new users

This script demonstrates how to:
1. Load a trained model
2. Prepare new user data
3. Make churn predictions
4. Identify high-risk users for targeted recommendations
"""

import pandas as pd
import numpy as np
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


def predict_churn_for_new_users():
    """Example of using the trained model for new predictions"""
    
    print("=" * 70)
    print("CHURN PREDICTION - EXAMPLE USAGE")
    print("=" * 70)
    
    # Step 1: Create sample new user data
    print("\n[Step 1] Creating sample new user data...")
    new_users = pd.DataFrame({
        'member_id': ['MEM999001', 'MEM999002', 'MEM999003', 'MEM999004', 'MEM999005'],
        'app_visits': [5, 25, 15, 2, 30],  # Low visits may indicate churn risk
        'web_visits': [2, 10, 8, 1, 15],
        'days_since_registration': [10, 365, 180, 5, 730],
        'total_purchases': [0, 10, 5, 0, 15],
        'avg_purchase_value': [0, 75, 50, 0, 100],
        'customer_service_calls': [5, 1, 2, 8, 0],  # High calls may indicate issues
        'account_age_days': [30, 400, 200, 15, 750],
        'platform_preference': ['iOS', 'Android', 'Web', 'iOS', 'Android'],
        'subscription_type': ['Free', 'Premium', 'Basic', 'Free', 'Premium']
    })
    
    print(f"New users to predict: {len(new_users)}")
    print("\nUser data:")
    print(new_users[['member_id', 'app_visits', 'web_visits', 'total_purchases', 'subscription_type']])
    
    # Step 2: Load the trained model
    print("\n[Step 2] Loading trained model...")
    trainer = ModelTraining()
    try:
        trainer.load_model('models/churn_model.pkl')
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("✗ Model not found. Please run main.py first to train the model.")
        return
    
    # Step 3: Prepare the data (same preprocessing as training)
    print("\n[Step 3] Preparing new user data...")
    data_prep = DataPreparation()
    
    # Store member IDs for later
    member_ids = new_users['member_id'].values
    
    # Remove member_id for prediction
    X_new = new_users.drop(columns=['member_id'])
    
    # Identify categorical features
    data_prep.categorical_features = X_new.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical features
    X_new = data_prep.encode_categorical_features(X_new)
    
    # Normalize features using the same scaler
    # Note: In production, you should save and load the scaler from training
    X_new_norm = data_prep.normalize_features(X_new)
    
    print("✓ Data preparation complete")
    
    # Step 4: Make predictions
    print("\n[Step 4] Making churn predictions...")
    churn_probabilities = trainer.predict_proba(X_new_norm)[:, 1]
    churn_predictions = trainer.predict(X_new_norm)
    
    # Step 5: Display results
    print("\n[Step 5] Prediction Results")
    print("-" * 70)
    
    results_df = pd.DataFrame({
        'Member ID': member_ids,
        'Churn Probability': churn_probabilities,
        'Predicted Churn': ['Yes' if x == 1 else 'No' for x in churn_predictions],
        'Risk Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                      for p in churn_probabilities]
    })
    
    print(results_df.to_string(index=False))
    
    # Step 6: Identify high-risk users for recommendations
    print("\n[Step 6] Targeted Recommendations")
    print("-" * 70)
    
    high_risk_users = results_df[results_df['Risk Level'] == 'High']
    medium_risk_users = results_df[results_df['Risk Level'] == 'Medium']
    
    if len(high_risk_users) > 0:
        print("\n🔴 HIGH RISK USERS - Immediate Action Required:")
        print(high_risk_users[['Member ID', 'Churn Probability']].to_string(index=False))
        print("\nRecommended Actions:")
        print("  • Send personalized retention offers")
        print("  • Contact via customer success team")
        print("  • Offer premium features trial")
        print("  • Investigate pain points")
    
    if len(medium_risk_users) > 0:
        print("\n🟡 MEDIUM RISK USERS - Monitor and Engage:")
        print(medium_risk_users[['Member ID', 'Churn Probability']].to_string(index=False))
        print("\nRecommended Actions:")
        print("  • Send engagement campaigns")
        print("  • Provide helpful resources/tutorials")
        print("  • Collect feedback")
    
    low_risk_users = results_df[results_df['Risk Level'] == 'Low']
    if len(low_risk_users) > 0:
        print("\n🟢 LOW RISK USERS - Maintain Engagement:")
        print(f"  {len(low_risk_users)} users with low churn risk")
        print("\nRecommended Actions:")
        print("  • Continue standard engagement")
        print("  • Encourage referrals")
        print("  • Upsell opportunities")
    
    print("\n" + "=" * 70)
    print("PREDICTION AND RECOMMENDATIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        predict_churn_for_new_users()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
