"""
Data preprocessing for stroke prediction.
This script downloads and preprocesses the stroke prediction dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import urllib.request
import zipfile

# Define paths
DATA_RAW_DIR = os.path.join('data', 'raw')
DATA_PROCESSED_DIR = os.path.join('data', 'processed')
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/stroke-prediction-dataset.zip"
DATASET_PATH = os.path.join(DATA_RAW_DIR, 'healthcare-dataset-stroke-data.csv')

def download_dataset():
    """Download the stroke dataset if it doesn't exist."""
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset...")
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        zip_path = os.path.join(DATA_RAW_DIR, 'dataset.zip')
        
        # Try to download from the specified URL
        try:
            urllib.request.urlretrieve(DATASET_URL, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_RAW_DIR)
            print("Dataset downloaded and extracted.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please manually download the dataset from: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset")
            print(f"and place it in {DATASET_PATH}")
            return False
        
        # Clean up
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    else:
        print(f"Dataset already exists at {DATASET_PATH}")
    
    return True

def load_data():
    """Load the stroke dataset."""
    if not os.path.exists(DATASET_PATH):
        if not download_dataset():
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """Explore the dataset and print summary statistics."""
    print("\nDataset Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df['stroke'].value_counts())
    print(f"Stroke rate: {df['stroke'].mean():.2%}")
    
    # Check for missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\nPreprocessing data...")
    
    # Handle 'Unknown' in smoking_status
    df['smoking_status'].replace('Unknown', np.nan, inplace=True)
    
    # Drop id column
    df = df.drop('id', axis=1, errors='ignore')
    
    # Define feature types
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split the data
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit the preprocessor on the training data
    print("Fitting preprocessing pipeline...")
    preprocessor.fit(X_train)
    
    # Save the preprocessor
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(DATA_PROCESSED_DIR, 'preprocessor.joblib'))
    
    # Transform the data
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the processed data
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_train.npy'), X_train_processed)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'X_test.npy'), X_test_processed)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(DATA_PROCESSED_DIR, 'y_test.npy'), y_test.values)
    
    # Save feature names and indices for interpretation
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # Get feature names from one-hot encoder
            encoder = transformer.named_steps['onehot']
            transformed_feature_names = encoder.get_feature_names_out(features)
            feature_names.extend(transformed_feature_names.tolist())
        else:
            feature_names.extend(features)
    
    joblib.dump(feature_names, os.path.join(DATA_PROCESSED_DIR, 'feature_names.joblib'))
    
    print(f"Processed data saved to {DATA_PROCESSED_DIR}")
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Testing data shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test

def main():
    """Main function to run the preprocessing pipeline."""
    # Create directories if they don't exist
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # Load the data
    df = load_data()
    
    # Explore the data
    explore_data(df)
    
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main() 