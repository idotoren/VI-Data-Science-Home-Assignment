"""
Model Evaluation Module for Churn Prediction System
Handles model evaluation metrics and performance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluation:
    """
    Handles model evaluation and performance metrics
    """
    
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive model evaluation"""
        print("\n=== Model Evaluation ===\n")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Add ROC-AUC if probabilities are provided
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            self.metrics['roc_auc'] = roc_auc
        
        # Print metrics
        print("Performance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if 'roc_auc' in self.metrics:
            print(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n=== Classification Report ===\n")
        print(classification_report(y_true, y_pred, target_names=['Not Churned', 'Churned']))
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to {save_path}")
        else:
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(f"\nConfusion matrix saved to confusion_matrix.png")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        else:
            plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to roc_curve.png")
        
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df, top_n=15, save_path=None):
        """Plot feature importance"""
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to feature_importance.png")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to {save_path}")
        else:
            plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
            print(f"Precision-recall curve saved to precision_recall_curve.png")
        
        plt.close()
    
    def generate_evaluation_report(self, y_true, y_pred, y_pred_proba, 
                                  feature_importance_df=None, output_dir='./'):
        """Generate complete evaluation report with all visualizations"""
        print("\n=== Generating Evaluation Report ===\n")
        
        # Evaluate model
        self.evaluate_model(y_true, y_pred, y_pred_proba)
        
        # Print classification report
        self.print_classification_report(y_true, y_pred)
        
        # Generate plots
        self.plot_confusion_matrix(y_true, y_pred, 
                                  save_path=f'{output_dir}/confusion_matrix.png')
        self.plot_roc_curve(y_true, y_pred_proba, 
                           save_path=f'{output_dir}/roc_curve.png')
        self.plot_precision_recall_curve(y_true, y_pred_proba,
                                        save_path=f'{output_dir}/precision_recall_curve.png')
        
        if feature_importance_df is not None:
            self.plot_feature_importance(feature_importance_df,
                                        save_path=f'{output_dir}/feature_importance.png')
        
        print("\n=== Evaluation Report Complete ===")
        return self.metrics
