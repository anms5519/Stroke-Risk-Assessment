"""
Evaluation script for stroke prediction model.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, auc, average_precision_score
)
import tensorflow as tf
import shap

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import StrokeModel

# Define paths
DATA_PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')
NOTEBOOKS_DIR = os.path.join('notebooks')

def load_model_and_data():
    """Load the trained model and test data."""
    # Load model
    model_path = os.path.join(MODELS_DIR, 'stroke_model.h5')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return None, None, None
    
    model = StrokeModel.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Load test data
    X_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_test.npy'))
    
    # Load feature names
    feature_names_path = os.path.join(DATA_PROCESSED_DIR, 'feature_names.joblib')
    try:
        feature_names = joblib.load(feature_names_path)
    except:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
    
    print(f"Test data loaded: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return model, X_test, y_test, feature_names

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and print metrics."""
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = model.predict_classes(X_test)
    
    # Print classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    # Plot precision-recall curve
    plot_precision_recall_curve(y_test, y_pred_proba)
    
    return y_pred_proba, y_pred

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'confusion_matrix.png'))
    plt.close()
    print("Confusion matrix saved to notebooks/confusion_matrix.png")

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'roc_curve.png'))
    plt.close()
    print("ROC curve saved to notebooks/roc_curve.png")
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_proba):
    """Plot precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid()
    
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'precision_recall_curve.png'))
    plt.close()
    print("Precision-Recall curve saved to notebooks/precision_recall_curve.png")
    
    return avg_precision

def find_optimal_threshold(y_true, y_pred_proba):
    """Find the optimal threshold for classification."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate the J statistic (Youden's J = Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"At this threshold - Sensitivity: {tpr[optimal_idx]:.4f}, Specificity: {1-fpr[optimal_idx]:.4f}")
    
    # Plot different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr[:-1], label='Sensitivity (True Positive Rate)')
    plt.plot(thresholds, 1-fpr[:-1], label='Specificity (1 - False Positive Rate)')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Sensitivity and Specificity vs Threshold')
    plt.legend()
    plt.grid()
    
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'threshold_analysis.png'))
    plt.close()
    print("Threshold analysis saved to notebooks/threshold_analysis.png")
    
    return optimal_threshold

def analyze_feature_importance(model, X_test, feature_names):
    """Analyze feature importance using SHAP values."""
    try:
        # Create a background dataset for SHAP
        background = X_test[np.random.choice(X_test.shape[0], 100, replace=False)]
        
        # Create explainer
        explainer = shap.DeepExplainer(
            model.model, 
            background
        )
        
        # Calculate SHAP values for the first 100 samples
        sample_size = min(100, X_test.shape[0])
        shap_values = explainer.shap_values(X_test[:sample_size])
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[0], X_test[:sample_size], feature_names=feature_names,
                          show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(NOTEBOOKS_DIR, 'shap_summary.png'))
        plt.close()
        print("SHAP summary plot saved to notebooks/shap_summary.png")
        
        # Bar plot of mean absolute SHAP values
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[0], X_test[:sample_size], feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(NOTEBOOKS_DIR, 'shap_bar.png'))
        plt.close()
        print("SHAP bar plot saved to notebooks/shap_bar.png")
        
    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        print("Skipping feature importance analysis.")

def main():
    """Main function to evaluate the model."""
    print("Starting model evaluation...")
    
    # Load the model and data
    model, X_test, y_test, feature_names = load_model_and_data()
    if model is None:
        return
    
    # Evaluate the model
    y_pred_proba, y_pred = evaluate_model(model, X_test, y_test)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    
    # Apply optimal threshold and re-evaluate
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    print("\nClassification Report with Optimal Threshold:")
    print(classification_report(y_test, y_pred_optimal))
    
    # Analyze feature importance
    analyze_feature_importance(model, X_test, feature_names)
    
    print("Model evaluation completed successfully!")

if __name__ == "__main__":
    main() 