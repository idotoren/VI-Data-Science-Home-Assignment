import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_auc_score, roc_curve, precision_recall_curve,
        log_loss, precision_score, recall_score, f1_score,
        average_precision_score, accuracy_score
    )

class ModelEvaluation:
    def __init__(self, score_column='score', label_column='churn',
                 marginal_cost=1, churn_prevention_rev=10, threshold=0.5, output_dir='Outputs'):
        self.score_column = score_column
        self.label_column = label_column
        self.marginal_cost = marginal_cost
        self.churn_prevention_rev = churn_prevention_rev
        self.threshold = threshold
        self.output_dir = output_dir


    def _evaluate_model(self, predictions_df):
        """Comprehensive model evaluation"""
        print("\n=== Model Evaluation ===\n")

        y_true = predictions_df.churn
        y_pred = predictions_df.score > self.threshold
        y_pred_proba = predictions_df.score if 'score' in predictions_df else None

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        logloss = log_loss(y_true, y_pred_proba)
        # precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        # recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        metrics = {
            'accuracy': accuracy,
            # 'precision': precision,
            # 'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

        # Print metrics
        print("Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  log_loss:  {logloss:.4f}")
        # print(f"  Precision: {precision:.4f}")
        # print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:   {metrics['pr_auc']:.4f}")

        return metrics

    def _plot_roc_curve(self, predictions_df, output_file='roc_curve.png'):
        """
        Plots the ROC curve and saves it to a PNG file.

        Args:
            y_true (array-like): True binary labels.
            y_pred_proba (array-like): Predicted probabilities for the positive class.
            output_file (str): Path to save the ROC curve PNG file.
        """
        y_true = predictions_df[self.label_column]
        y_pred_proba = predictions_df[self.score_column]
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_file))
        print(f"ROC curve saved to {output_file}")

    def _plot_qq_calibration(self, df, bins=100, figsize=(8, 6)):
        """
        Plots a Q-Q plot comparing estimated probabilities vs actual labels.
        Points are binned based on score, and point size is based on density.

        Parameters:
            df (pd.DataFrame): DataFrame with estimated probabilities and actual labels.
            score_col (str): Column name for estimated probabilities.
            label_col (str): Column name for actual labels (0/1).
            bins (int): Number of quantile bins (default: 100).
            figsize (tuple): Size of the plot.
        """
        df = df.copy()

        # Create equal-width bins between 0 and 1
        bin_edges = np.linspace(0, 1, bins + 1)
        df['prob_bin'] = pd.cut(df[self.score_column], bins=bin_edges, include_lowest=True, right=False)

        # Aggregate by bins
        bin_stats = df.groupby('prob_bin').agg(
            mean_score=(self.score_column, 'mean'),
            mean_label=(self.label_column, 'mean'),
            count=('prob_bin', 'count')
        ).reset_index()

        # Normalize counts for dot size
        max_count = bin_stats['count'].max()
        bin_stats['size'] = bin_stats['count'] / max_count * 300  # scale size

        # Plot
        plt.figure(figsize=figsize)
        plt.scatter(
            bin_stats['mean_score'],
            bin_stats['mean_label'],
            s=bin_stats['size'],
            alpha=0.6,
            edgecolor='k',
            linewidth=0.5
        )
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Estimated Probability (Mean per Bin)')
        plt.ylabel('Actual Churn Rate (Mean per Bin)')
        plt.title('Q-Q Plot: Predicted vs Actual (Binned)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'qq_plot.png'))
        plt.show()


    def _compute_calibration_by_probability_bin(self, df, bins=10):
        df = df.copy()

        # Create equal-width bins between 0 and 1
        bin_edges = np.linspace(0, 1, bins + 1)
        df['prob_bin'] = pd.cut(df[self.score_column], bins=bin_edges, include_lowest=True, right=False)

        # Group by bin and compute calibration stats
        calib = df.groupby('prob_bin').agg(
            sum_score=(self.score_column, 'sum'),
            sum_label=(self.label_column, 'sum'),
            count=(self.label_column, 'count')
        ).reset_index()

        # Avoid division by zero
        calib['calibration_ratio'] = calib['sum_score'] / calib['sum_label'].replace(0, np.nan)
        calib['calibration_ratio'] = calib['calibration_ratio'].fillna(np.nan)

        # Compute overall average calibration
        total_score = df[self.score_column].sum()
        total_label = df[self.label_column].sum()
        overall_calibration = total_score / total_label if total_label > 0 else np.nan

        print(f'Average Calibration Ratio (Overall): {overall_calibration:.4f}')

        return calib[['prob_bin', 'sum_score', 'sum_label', 'count', 'calibration_ratio']]


    def _prepare_recommendation_list(self, predictions_df, N, output_file='recommendations.csv'):
        """
        Prepare a recommendation list by filtering non-churned samples, sorting by score, and saving the top N results.

        Args:
            predictions_df (pd.DataFrame): DataFrame with columns 'member_id', 'score', and 'churn'.
            N (int): Number of top recommendations to include.
            output_file (str): Path to save the resulting CSV file.

        Returns:
            pd.DataFrame: DataFrame with 'member_id', 'score', and 'rank' for the top N recommendations.
        """
        # Filter out samples that have already churned
        filtered_df = predictions_df[predictions_df['churn'] == 0]

        # Sort by score in descending order
        sorted_df = filtered_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        # Add rank column
        sorted_df['rank'] = sorted_df.index + 1

        # Select the top N rows
        top_n_df = sorted_df.head(N)

        # Save to CSV
        top_n_df[['member_id', 'score', 'rank']].to_csv(os.path.join(self.output_dir,output_file), index=False)

        return top_n_df[['member_id', 'score', 'rank']]


    def run(self, predictions_df):

        """
        Complete model evaluation pipeline
        Args:
            predictions_df (pd.DataFrame): DataFrame with columns 'member_id', 'score', and 'churn'.
        Returns:
            dict: Dictionary with evaluation metrics.
        """
        print("\n=== Starting Model Evaluation and results preparation Pipeline ===\n")

        # Step 1: Evaluate model performance
        metrics = self._evaluate_model(predictions_df)

        # Step 2: Plot Q-Q calibration and auc curve
        self._plot_qq_calibration(predictions_df, bins=100)
        self._plot_roc_curve(predictions_df)

        # Step 3: Compute calibration by probability bin
        calib_df = self._compute_calibration_by_probability_bin(predictions_df, bins=10)
        print("\nCalibration by Probability Bin:")
        print(calib_df)

        # Step 4: Prepare recommendation list
        recommendations = self._prepare_recommendation_list(predictions_df, N=100, output_file='top_100_recommendations.csv')
        print(recommendations.head())
        print(f"\n=== Model Evaluation Complete ===")

        return metrics

