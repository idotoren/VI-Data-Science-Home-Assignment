from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import warnings
import pandas as pd
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')


class ModelTraining:
    """
    Handles model training, hyperparameter tuning, and prediction
    """

    def __init__(self, target_column='churn', member_id_column='member_id',
                 eval_mode=False):
        self.target_column = target_column
        self.member_id_column = member_id_column
        self.eval_mode = eval_mode
        self.best_model = None
        self.model_name = None
        self.models = {
            'LR': LogisticRegression(random_state=42, max_iter=5000),
            'RF': RandomForestClassifier(random_state=42, n_estimators=100),
            'LGBM': LGBMClassifier(boosting_type='gbdt', random_state=42, n_estimators=100,
                                       learning_rate=0.1, max_depth=-1)
        }


    def _tune_hyperparameters(self, x_train, y_train, model_name='LR'):
        """Tune hyperparameters using GridSearchCV with inner cross-validation."""
        print(f"\n=== Tuning Hyperparameters for {model_name} ===\n")

        if model_name == 'RF':
            param_grid = {
                'n_estimators': [50, 200, 400],
                'max_depth': [10, 20, None],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        elif model_name == 'LGBM':
            param_grid = {
                'n_estimators': [50, 200, 400],
                'learning_rate': [0.01, 0.1],
                'min_split_gain': [1, 3, 5]
            }
            base_model = LGBMClassifier(random_state=42, n_jobs=-1)

        else:  # Logistic Regression
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs']
            }
            base_model = LogisticRegression(random_state=42, max_iter=5000)

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(x_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best ROC-AUC score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_


    def _stratified_kfold_training(self, data, model_type, n_splits=10, random_state=42):
        """
        Perform stratified K-Fold cross-validation using required model, train the model, and create a prediction dataset.

        Args:
            data (pd.DataFrame): Input dataset with features, target, and member_id.
            model_type (str): Model type to use ('LR' for Logistic Regression, 'RF' for Random Forest, 'LGBM' for gradient boosting).
            n_splits (int): Number of folds for cross-validation.
            random_state (int): Random state for reproducibility.

        Returns:
            pd.DataFrame: Prediction dataset with member_id, score, rank, and target.
        """
        # Separate features and target
        x = data.drop(columns=[self.target_column, self.member_id_column])
        y = data[self.target_column]
        member_ids = data[self.member_id_column]

        # Initialize Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialize arrays to store OOF predictions and param tuning
        oof_scores = np.zeros(len(data))
        best_params_list = []

        print("\n=== Training Model ===\n")

        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
            print(f"Training fold {fold + 1}/{n_splits}...")

            # Split data into training and validation sets
            x_train, x_val = x.iloc[train_idx], x.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # validate supported model based on the model_type flag
            if model_type not in self.models.keys():
                raise ValueError("Invalid model_type. Choose 'LR' for Logistic Regression, 'LGBM' for lightGBM boosting or 'RF' for Random Forest.")

            # Tune hyperparameters and train
            best_params = self._tune_hyperparameters(x_train, y_train, model_type)
            best_params_list.append(best_params)

            # Train the model
            model = self.models[model_type].set_params(**best_params, random_state=42)
            model.fit(x_train, y_train)

            # Predict on the validation set
            oof_scores[val_idx] = model.predict_proba(x_val)[:, 1]  # Assuming binary classification

        # Create the prediction dataset
        prediction_df = pd.DataFrame({
            self.member_id_column: member_ids,
            'score': oof_scores,
            self.target_column: y
        })

        # Add rank based on scores
        prediction_df['rank'] = prediction_df['score'].rank(ascending=False, method='dense')

        return prediction_df

    def train(self, data, model_type='LR'):
        """
        Complete model training pipeline
        Args:
            data (pd.DataFrame): Input dataset with features, target, and member_id.
            model_type (str): Model type to use ('LR' for Logistic Regression, 'RF' for Random Forest, 'LGBM' for gradient boosting).
        Returns:
            pd.DataFrame: Prediction dataset with member_id, score, rank, and target.
        """
        print("\n=== Starting Model Training Pipeline ===\n")
        if self.eval_mode:
            print("Evaluation mode is ON: Model selection metrics will be printed.\n")
            results, model_type = self._compare_models_and_select()

        predictions_df = self._stratified_kfold_training(data, model_type)

        print(f"\n=== Model Training Pipeline Complete ===")
        print(f"Prediction dataset shape: {predictions_df.shape}")

        return predictions_df
