"""
Exploratory data analysis for the stroke prediction dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import (
    set_plotting_style, plot_feature_distributions, 
    plot_correlation_matrix
)

# Define paths
DATA_RAW_DIR = os.path.join('data', 'raw')
DATASET_PATH = os.path.join(DATA_RAW_DIR, 'healthcare-dataset-stroke-data.csv')
NOTEBOOKS_DIR = 'notebooks'

def load_data():
    """Load the stroke dataset."""
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please run preprocess.py first.")
        return None
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

def explore_data(df):
    """Explore dataset and print statistics."""
    # Basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df['stroke'].value_counts())
    print(f"Stroke rate: {df['stroke'].mean():.2%}")
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Check unique values in categorical columns
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    print("\nUnique values in categorical features:")
    for feature in categorical_features:
        print(f"{feature}: {df[feature].unique()}")
    
    return categorical_features

def analyze_feature_relationships(df, categorical_features):
    """Analyze relationships between features and target."""
    # Define numeric features
    numeric_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    
    # Plot distributions
    plot_feature_distributions(df, categorical_features, numeric_features)
    
    # Plot correlation matrix
    plot_correlation_matrix(df, numeric_features + ['stroke'])
    
    # Analyze age vs. stroke
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='stroke', y='age', data=df)
    plt.title('Age vs. Stroke')
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'age_vs_stroke.png'))
    plt.close()
    
    # Analyze glucose level vs. stroke
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
    plt.title('Average Glucose Level vs. Stroke')
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'glucose_vs_stroke.png'))
    plt.close()
    
    # Cross tabulation for categorical features
    for feature in categorical_features:
        cross_tab = pd.crosstab(df[feature], df['stroke'])
        cross_tab_pct = pd.crosstab(df[feature], df['stroke'], normalize='index') * 100
        
        print(f"\nCross-tabulation for {feature}:")
        print(cross_tab)
        print(f"\nPercentage for {feature}:")
        print(cross_tab_pct)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cross_tab_pct.index, y=cross_tab_pct[1])
        plt.title(f'Stroke Rate by {feature}')
        plt.xlabel(feature)
        plt.ylabel('Stroke Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(NOTEBOOKS_DIR, f'{feature}_vs_stroke.png'))
        plt.close()

def analyze_combined_risk_factors(df):
    """Analyze combined risk factors."""
    # Create a risk score based on hypertension and heart disease
    df['risk_score'] = df['hypertension'] + df['heart_disease']
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='risk_score', hue='stroke', data=df)
    plt.title('Stroke by Risk Score (Hypertension + Heart Disease)')
    plt.xlabel('Risk Score')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'risk_score_vs_stroke.png'))
    plt.close()
    
    # Calculate stroke rate by risk score
    risk_rates = df.groupby('risk_score')['stroke'].mean() * 100
    print("\nStroke Rate by Risk Score:")
    print(risk_rates)
    
    # Age groups analysis
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 40, 60, 80, 100], 
                             labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
    
    # Calculate stroke rate by age group
    age_rates = df.groupby('age_group')['stroke'].mean() * 100
    print("\nStroke Rate by Age Group:")
    print(age_rates)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=age_rates.index, y=age_rates.values)
    plt.title('Stroke Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Stroke Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'age_group_vs_stroke.png'))
    plt.close()
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], 
                               labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Calculate stroke rate by BMI category
    bmi_rates = df.groupby('bmi_category')['stroke'].mean() * 100
    print("\nStroke Rate by BMI Category:")
    print(bmi_rates)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=bmi_rates.index, y=bmi_rates.values)
    plt.title('Stroke Rate by BMI Category')
    plt.xlabel('BMI Category')
    plt.ylabel('Stroke Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(NOTEBOOKS_DIR, 'bmi_category_vs_stroke.png'))
    plt.close()

def main():
    """Main function to run exploratory data analysis."""
    print("Starting exploratory data analysis...")
    
    # Set plotting style
    set_plotting_style()
    
    # Create notebooks directory if it doesn't exist
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Basic exploration
    categorical_features = explore_data(df)
    
    # Analyze feature relationships
    analyze_feature_relationships(df, categorical_features)
    
    # Analyze combined risk factors
    analyze_combined_risk_factors(df)
    
    print("Exploratory data analysis completed successfully!")
    print(f"Plots saved to {NOTEBOOKS_DIR} directory.")

if __name__ == "__main__":
    main() 