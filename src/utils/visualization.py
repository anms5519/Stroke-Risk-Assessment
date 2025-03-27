"""
Visualization utilities for the stroke prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def set_plotting_style():
    """Set the plotting style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    sns.set_palette('viridis')
    return plt

def plot_feature_distributions(df, categorical_features, numeric_features, target='stroke', 
                              save_dir='notebooks'):
    """
    Plot the distributions of features by target.
    
    Args:
        df: Dataframe containing features and target
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        target: Target column name
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # Plot categorical features
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=feature, hue=target, data=df)
        plt.title(f'Distribution of {feature} by {target}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'dist_{feature}.png'))
        plt.close()
    
    # Plot numeric features
    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=feature, hue=target, kde=True, element='step', 
                     common_norm=False, palette='viridis')
        plt.title(f'Distribution of {feature} by {target}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'dist_{feature}.png'))
        plt.close()

def plot_correlation_matrix(df, numeric_features, save_dir='notebooks'):
    """
    Plot correlation matrix for numeric features.
    
    Args:
        df: Dataframe containing features
        numeric_features: List of numeric feature names
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # Calculate correlation matrix
    corr = df[numeric_features].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, annot=True, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    plt.close()

def plot_learning_curves(history, save_dir='notebooks'):
    """
    Plot learning curves from training history.
    
    Args:
        history: Training history from model.fit
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_prediction_distribution(y_pred_proba, y_true, save_dir='notebooks'):
    """
    Plot distribution of prediction probabilities by true class.
    
    Args:
        y_pred_proba: Predicted probabilities
        y_true: True labels
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    set_plotting_style()
    
    # Create dataframe for plotting
    pred_df = pd.DataFrame({
        'Probability': y_pred_proba.flatten(),
        'True_Class': y_true.flatten()
    })
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pred_df, x='Probability', hue='True_Class', 
                 element='step', stat='density', common_norm=False, bins=30)
    plt.title('Distribution of Prediction Probabilities by True Class')
    plt.xlabel('Predicted Probability of Stroke')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_distribution.png'))
    plt.close()
    
    return pred_df 