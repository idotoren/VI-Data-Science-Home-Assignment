"""
Model Training Module for Churn Prediction System
Handles model selection, training, and hyperparameter tuning
"""

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


class ModelTraining:
    """
    Handles model training, hyperparameter tuning, and model selection
    """

    def __init__(self):
        self.best_model = None
        self.model_name = None
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test

    def compare_models(self, X_train, y_train, cv=5):
        """Compare different models using cross-validation"""
        print("\n=== Comparing Models ===\n")

        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            results[name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std()
            }
            print(f"  ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['mean_score'])
        print(f"\nBest model: {best_model_name}")
        return results, best_model_name

    def tune_hyperparameters(self, X_train, y_train, model_name='Random Forest'):
        """Tune hyperparameters using GridSearchCV"""
        print(f"\n=== Tuning Hyperparameters for {model_name} ===\n")

        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
            base_model = GradientBoostingClassifier(random_state=42)

        else:  # Logistic Regression
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
            base_model = LogisticRegression(random_state=42, max_iter=1000)

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_model(self, X_train, y_train, model_name='Random Forest', tune=True):
        """Train the final model"""
        print("\n=== Training Final Model ===\n")

        if tune:
            # Tune hyperparameters and train
            self.best_model = self.tune_hyperparameters(X_train, y_train, model_name)
        else:
            # Use default model
            self.best_model = self.models[model_name]
            self.best_model.fit(X_train, y_train)

        self.model_name = model_name
        print(f"\n{model_name} training complete!")
        return self.best_model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_model first.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_model first.")
        return self.best_model.predict_proba(X)

    def save_model(self, filepath):
        """Save the trained model to disk"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_model first.")

        joblib.dump(self.best_model, filepath)
        print(f"\nModel saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from disk"""
        self.best_model = joblib.load(filepath)
        print(f"\nModel loaded from {filepath}")
        return self.best_model

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np

    def stratified_kfold_training(data, target_column, member_id_column, model_type='LR', n_splits=5, random_state=42):
        """
        Perform stratified K-Fold cross-validation using Logistic Regression or Random Forest, train the model, and create a prediction dataset.

        Args:
            data (pd.DataFrame): Input dataset with features, target, and member_id.
            target_column (str): Name of the target column.
            member_id_column (str): Name of the member_id column.
            model_type (str): Model type to use ('LR' for Logistic Regression, 'RF' for Random Forest).
            n_splits (int): Number of folds for cross-validation.
            random_state (int): Random state for reproducibility.

        Returns:
            pd.DataFrame: Prediction dataset with member_id, score, rank, and target.
        """
        # Separate features and target
        X = data.drop(columns=[target_column, member_id_column])
        y = data[target_column]
        member_ids = data[member_id_column]

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialize arrays to store OOF predictions
        oof_scores = np.zeros(len(data))

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Training fold {fold + 1}/{n_splits}...")

            # Split data into training and validation sets
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Initialize the model based on the model_type flag
            if model_type == 'LR':
                model = LogisticRegression(random_state=random_state, max_iter=1000)
            elif model_type == 'RF':
                model = RandomForestClassifier(random_state=random_state)
            else:
                raise ValueError("Invalid model_type. Choose 'LR' for Logistic Regression or 'RF' for Random Forest.")

            # Train the model
            model.fit(X_train, y_train)

            # Predict on the validation set
            oof_scores[val_idx] = model.predict_proba(X_val)[:, 1]  # Assuming binary classification

        # Create the prediction dataset
        prediction_df = pd.DataFrame({
            member_id_column: member_ids,
            'score': oof_scores,
            target_column: y
        })

        # Add rank based on scores
        prediction_df['rank'] = prediction_df['score'].rank(ascending=False, method='dense')

        return prediction_df

    # Example usage
    # prediction_dataset = stratified_kfold_training(data, target_column='churn', member_id_column='member_id', model_type='RF')
    # print(prediction_dataset.head())