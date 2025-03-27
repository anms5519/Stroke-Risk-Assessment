"""
Training script for stroke prediction model.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import StrokeModel, calculate_class_weights

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_PROCESSED_DIR = os.path.join('data', 'processed')
MODELS_DIR = os.path.join('models')

def load_processed_data():
    """Load the preprocessed data."""
    X_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(DATA_PROCESSED_DIR, 'y_test.npy'))
    
    print(f"Loaded training data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Loaded test data: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """Train the stroke prediction model."""
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Create the model
    input_dim = X_train.shape[1]
    model = StrokeModel(
        input_dim=input_dim,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        l2_reg=0.001,
        learning_rate=0.001,
        class_weight=class_weights
    )
    
    # Print model summary
    model.model.summary()
    
    # Create a TensorBoard callback
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    
    # Create model checkpoint callback
    os.makedirs(MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(MODELS_DIR, 'model_checkpoint.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_auc',
        mode='max'
    )
    
    # Split training data into training and validation sets
    val_split = 0.2
    split_idx = int(X_train.shape[0] * (1 - val_split))
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # Train the model
    history = model.fit(
        X_train_split, y_train_split,
        X_val=X_val, y_val=y_val,
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )
    
    # Save the final model
    final_model_path = os.path.join(MODELS_DIR, 'stroke_model.h5')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def cross_validate(X_train, y_train, n_splits=5):
    """Perform cross-validation on the training data."""
    print("\nPerforming cross-validation...")
    
    # Initialize K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Metrics for each fold
    fold_metrics = {
        'loss': [],
        'accuracy': [],
        'auc': [],
        'precision': [],
        'recall': []
    }
    
    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining fold {fold+1}/{n_splits}")
        
        # Split data
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Calculate class weights
        class_weights = calculate_class_weights(y_fold_train)
        
        # Create and train model
        model = StrokeModel(
            input_dim=X_train.shape[1],
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            l2_reg=0.001,
            learning_rate=0.001,
            class_weight=class_weights
        )
        
        # Train
        model.fit(
            X_fold_train, y_fold_train,
            X_val=X_fold_val, y_val=y_fold_val,
            epochs=30,  # Fewer epochs for CV
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        metrics = model.evaluate(X_fold_val, y_fold_val)
        
        # Store metrics
        for metric, value in metrics.items():
            fold_metrics[metric].append(value)
        
        print(f"Fold {fold+1} metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Average metrics
    print("\nCross-validation results:")
    for metric, values in fold_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

def plot_training_history(history):
    """Plot the training history."""
    os.makedirs('notebooks', exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy metrics
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model Accuracy and AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join('notebooks', 'training_history.png'))
    print("Training history plot saved to notebooks/training_history.png")

def main():
    """Main function to train the model."""
    print("Starting model training...")
    
    # Check if processed data exists
    if not os.path.exists(os.path.join(DATA_PROCESSED_DIR, 'X_train.npy')):
        print("Processed data not found. Please run preprocess.py first.")
        return
    
    # Load the processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Cross-validation
    cross_validate(X_train, y_train, n_splits=5)
    
    # Train the final model
    model, history = train_model(X_train, y_train, X_test, y_test)
    
    print("Model training completed successfully!")

if __name__ == "__main__":
    main() 