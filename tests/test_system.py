"""
Basic tests for the churn prediction system components
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining
from src.model_evaluation import ModelEvaluation
from src.generate_data import generate_sample_data


def test_data_generation():
    """Test that data generation works correctly"""
    print("\n=== Testing Data Generation ===")
    
    # Generate small sample
    df = generate_sample_data(n_samples=100, output_path='/tmp/test_data.csv')
    
    assert df is not None, "DataFrame should not be None"
    assert len(df) >= 100, "Should generate at least 100 samples"
    assert 'churn' in df.columns, "Should have churn column"
    assert 'member_id' in df.columns, "Should have member_id column"
    
    print("✓ Data generation test passed")
    return True


def test_data_preparation():
    """Test data preparation pipeline"""
    print("\n=== Testing Data Preparation ===")
    
    # Generate test data
    df = generate_sample_data(n_samples=100, output_path='/tmp/test_data.csv')
    
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Test data preparation
    X, y = data_prep.prepare_data(df, target_column='churn')
    
    assert X is not None, "Features should not be None"
    assert y is not None, "Target should not be None"
    assert len(X) == len(y), "Features and target should have same length"
    assert 'member_id' not in X.columns, "member_id should be removed"
    assert X.isnull().sum().sum() == 0, "Should have no missing values"
    
    print("✓ Data preparation test passed")
    return True


def test_model_training():
    """Test model training pipeline"""
    print("\n=== Testing Model Training ===")
    
    # Generate test data
    df = generate_sample_data(n_samples=200, output_path='/tmp/test_data.csv')
    
    # Prepare data
    data_prep = DataPreparation()
    X, y = data_prep.prepare_data(df, target_column='churn')
    
    # Initialize trainer
    trainer = ModelTraining()
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    
    assert len(X_train) > len(X_test), "Training set should be larger"
    
    # Normalize
    X_train_norm, X_test_norm = data_prep.normalize_features(X_train, X_test)
    
    # Train model (without tuning for speed)
    model = trainer.train_model(X_train_norm, y_train, 
                               model_name='Logistic Regression', 
                               tune=False)
    
    assert model is not None, "Model should be trained"
    
    # Make predictions
    y_pred = trainer.predict(X_test_norm)
    
    assert len(y_pred) == len(y_test), "Predictions should match test set size"
    
    print("✓ Model training test passed")
    return True


def test_model_evaluation():
    """Test model evaluation"""
    print("\n=== Testing Model Evaluation ===")
    
    # Generate test data
    df = generate_sample_data(n_samples=150, output_path='/tmp/test_data.csv')
    
    # Prepare data
    data_prep = DataPreparation()
    X, y = data_prep.prepare_data(df, target_column='churn')
    
    # Train model
    trainer = ModelTraining()
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
    X_train_norm, X_test_norm = data_prep.normalize_features(X_train, X_test)
    model = trainer.train_model(X_train_norm, y_train, 
                               model_name='Logistic Regression', 
                               tune=False)
    
    # Evaluate
    evaluator = ModelEvaluation()
    y_pred = trainer.predict(X_test_norm)
    y_pred_proba = trainer.predict_proba(X_test_norm)
    
    metrics = evaluator.evaluate_model(y_test, y_pred, y_pred_proba)
    
    assert 'accuracy' in metrics, "Should have accuracy metric"
    assert 'precision' in metrics, "Should have precision metric"
    assert 'recall' in metrics, "Should have recall metric"
    assert 'f1_score' in metrics, "Should have f1_score metric"
    assert 'roc_auc' in metrics, "Should have roc_auc metric"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be between 0 and 1"
    
    print("✓ Model evaluation test passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("RUNNING TESTS FOR CHURN PREDICTION SYSTEM")
    print("=" * 70)
    
    try:
        test_data_generation()
        test_data_preparation()
        test_model_training()
        test_model_evaluation()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {str(e)}")
        return False
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
