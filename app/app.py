"""
Flask web application for stroke prediction.
"""

import os
import sys
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import StrokeModel

# Define paths
MODEL_PATH = os.path.join('models', 'stroke_model.pkl')
PREPROCESSOR_PATH = os.path.join('data', 'processed', 'preprocessor.joblib')

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = None
preprocessor = None

def load_model_and_preprocessor():
    """Load the model and preprocessor."""
    global model, preprocessor
    
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found at {MODEL_PATH}. Using a mock model.")
            model = StrokeModel()  # Create a mock model
        else:
            # Load model
            model = StrokeModel.load(MODEL_PATH)
        
        # Check if preprocessor exists
        if not os.path.exists(PREPROCESSOR_PATH):
            print(f"Preprocessor not found at {PREPROCESSOR_PATH}. Using StandardScaler.")
            # Create a simple preprocessor
            from sklearn.preprocessing import StandardScaler
            preprocessor = StandardScaler()
        else:
            # Load preprocessor
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        print("Model and preprocessor setup complete.")
    except Exception as e:
        print(f"Error loading model or preprocessor: {str(e)}")
        # Use dummy model and preprocessor for development/testing
        from sklearn.preprocessing import StandardScaler
        model = StrokeModel()
        preprocessor = StandardScaler()

# Load model at startup
load_model_and_preprocessor()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

def get_feature_importance(input_data):
    """Calculate simplified feature importance based on known medical risk factors."""
    feature_importance = []
    
    # Age - major risk factor
    age = input_data.get('Age', 0)
    if age > 65:
        feature_importance.append(('Age', 0.9))
    elif age > 55:
        feature_importance.append(('Age', 0.7))
    elif age > 45:
        feature_importance.append(('Age', 0.4))
    else:
        feature_importance.append(('Age', 0.2))
    
    # Hypertension - major risk factor
    if input_data.get('Hypertension', 0) == 1:
        feature_importance.append(('Hypertension', 0.8))
    
    # Heart Disease - major risk factor
    if input_data.get('Heart_Disease', 0) == 1:
        feature_importance.append(('Heart Disease', 0.75))
    
    # Glucose level - major risk factor for diabetes which increases stroke risk
    glucose = input_data.get('Avg_Glucose_Level', 0)
    if glucose > 180:
        feature_importance.append(('Blood Glucose', 0.85))
    elif glucose > 140:
        feature_importance.append(('Blood Glucose', 0.65))
    elif glucose > 100:
        feature_importance.append(('Blood Glucose', 0.3))
    else:
        feature_importance.append(('Blood Glucose', 0.1))
    
    # BMI - moderate risk factor
    bmi = input_data.get('BMI', 0)
    if bmi > 35:  # Severely obese
        feature_importance.append(('BMI', 0.7))
    elif bmi > 30:  # Obese
        feature_importance.append(('BMI', 0.5))
    elif bmi > 25:  # Overweight
        feature_importance.append(('BMI', 0.3))
    else:
        feature_importance.append(('BMI', 0.1))
    
    # Smoking - significant risk factor
    smoking = input_data.get('Smoking_Status', '')
    if smoking == 'smokes':
        feature_importance.append(('Smoking', 0.75))
    elif smoking == 'formerly smoked':
        feature_importance.append(('Smoking', 0.5))
    else:
        feature_importance.append(('Smoking', 0.1))
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance[:5]  # Return top 5 factors

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on form data."""
    try:
        # Get form data
        form_data = {
            'gender': request.form['gender'],
            'Age': float(request.form['age']),
            'Hypertension': int(request.form['hypertension']),
            'Heart_Disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['residence_type'],
            'Avg_Glucose_Level': float(request.form['avg_glucose_level']),
            'BMI': float(request.form['bmi']),
            'Smoking_Status': request.form['smoking_status']
        }
        
        # Mock prediction based on risk factors
        age_factor = min(1.0, form_data['Age'] / 100)
        hypertension_factor = form_data['Hypertension'] * 0.3
        heart_disease_factor = form_data['Heart_Disease'] * 0.3
        glucose_factor = min(1.0, form_data['Avg_Glucose_Level'] / 300) * 0.2
        bmi_factor = min(1.0, (form_data['BMI'] - 18) / 30) * 0.2
        smoking_factor = 0.3 if form_data['Smoking_Status'] == 'smokes' else 0.15 if form_data['Smoking_Status'] == 'formerly smoked' else 0
        
        # Combine factors
        prediction_proba = (age_factor + hypertension_factor + heart_disease_factor + glucose_factor + bmi_factor + smoking_factor) / 2
        prediction_proba = max(0.05, min(0.95, prediction_proba))  # Limit to reasonable range
        
        # Get feature importance
        key_factors = get_feature_importance(form_data)
        
        # Return results
        return render_template(
            'result.html',
            prediction=prediction_proba,
            input_data=form_data,
            key_factors=key_factors
        )
    
    except Exception as e:
        error_title = "Input Processing Error"
        error_message = f"We couldn't process your health information: {str(e)}"
        return render_template('error.html', error_title=error_title, error_message=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction."""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Mock prediction based on risk factors
        age_factor = min(1.0, data.get('Age', 0) / 100)
        hypertension_factor = data.get('Hypertension', 0) * 0.3
        heart_disease_factor = data.get('Heart_Disease', 0) * 0.3
        glucose_factor = min(1.0, data.get('Avg_Glucose_Level', 0) / 300) * 0.2
        bmi_factor = min(1.0, (data.get('BMI', 20) - 18) / 30) * 0.2
        smoking_status = data.get('Smoking_Status', '').lower()
        smoking_factor = 0.3 if 'smoke' in smoking_status and 'never' not in smoking_status else 0
        
        # Combine factors
        prediction_proba = (age_factor + hypertension_factor + heart_disease_factor + glucose_factor + bmi_factor + smoking_factor) / 2
        prediction = 1 if prediction_proba >= 0.5 else 0
        
        # Get feature importance
        key_factors = get_feature_importance(data)
        
        # Return JSON response
        return jsonify({
            'prediction': prediction,
            'probability': prediction_proba,
            'risk_level': 'High' if prediction_proba >= 0.5 else 'Moderate' if prediction_proba >= 0.2 else 'Low',
            'key_factors': key_factors
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs(os.path.join('app', 'templates'), exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000) 