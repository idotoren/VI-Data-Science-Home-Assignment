"""
Main Pipeline for Churn Prediction System

This script orchestrates the entire pipeline:
1. Data Preparation (cleaning, encoding, normalization)
2. Model Training (with hyperparameter tuning)
3. Model Evaluation (metrics and visualizations)

Usage:
    python main.py
"""

import os
import sys
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation
from src.generate_data import generate_sample_data


def main():
    """Main pipeline execution"""
    
    print("=" * 70)
    print("CHURN PREDICTION SYSTEM - RECOMMENDATION ENGINE")
    print("=" * 70)
    
    # Configuration
    data_path = 'data/customer_data.csv'
    model_path = 'models/churn_model.pkl'
    output_dir = 'outputs'
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate sample data (if not exists)
    if not os.path.exists(data_path):
        print("\n[Step 1] Generating sample dataset...")
        generate_sample_data(n_samples=1000, output_path=data_path)
    else:
        print(f"\n[Step 1] Using existing dataset: {data_path}")
    
    # Step 2: Data Preparation
    print("\n[Step 2] Data Preparation")
    print("-" * 70)
    data_prep = DataPreparation()
    
    # Load data
    df = data_prep.load_data(data_path)
    
    # Prepare data (cleaning, encoding, etc.)
    X, y = data_prep.prepare_data(df, target_column='churn')
    
    # Step 3: Model Training
    print("\n[Step 3] Model Training")
    print("-" * 70)
    trainer = ModelTraining()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    
    # Normalize features
    X_train_norm, X_test_norm = data_prep.normalize_features(X_train, X_test)
    
    # Compare models (optional - comment out to skip)
    print("\n[Step 3a] Comparing different models...")
    model_results, best_model_name = trainer.compare_models(X_train_norm, y_train, cv=3)
    
    # Train final model with hyperparameter tuning
    print(f"\n[Step 3b] Training final model: {best_model_name}")
    model = trainer.train_model(X_train_norm, y_train, 
                               model_name=best_model_name, 
                               tune=True)
    
    # Calculate feature importance
    feature_importance_df = data_prep.calculate_feature_importance(X_train_norm, y_train, top_n=10)
    
    # Step 4: Model Evaluation
    print("\n[Step 4] Model Evaluation")
    print("-" * 70)
    evaluator = ModelEvaluation()
    
    # Make predictions
    y_pred = trainer.predict(X_test_norm)
    y_pred_proba = trainer.predict_proba(X_test_norm)
    
    # Generate evaluation report
    metrics = evaluator.generate_evaluation_report(
        y_test, 
        y_pred, 
        y_pred_proba,
        feature_importance_df=feature_importance_df,
        output_dir=output_dir
    )
    
    # Step 5: Save model
    print("\n[Step 5] Saving model")
    print("-" * 70)
    trainer.save_model(model_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\nModel Performance Summary:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nOutputs:")
    print(f"  Model saved to: {model_path}")
    print(f"  Visualizations saved to: {output_dir}/")
    print(f"    - confusion_matrix.png")
    print(f"    - roc_curve.png")
    print(f"    - precision_recall_curve.png")
    print(f"    - feature_importance.png")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM READY FOR DEPLOYMENT")
    print("=" * 70)
    
    # Identify high-risk users (example)
    print("\n[Bonus] Identifying High-Risk Users for Recommendations...")
    print("-" * 70)
    
    # Get churn probabilities for test set
    churn_probs = y_pred_proba[:, 1]
    high_risk_threshold = 0.7
    high_risk_users = (churn_probs >= high_risk_threshold).sum()
    
    print(f"Users with churn probability >= {high_risk_threshold}: {high_risk_users}")
    print(f"Percentage of high-risk users: {high_risk_users/len(y_test):.2%}")
    print("\nThese users should be targeted with retention campaigns!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
