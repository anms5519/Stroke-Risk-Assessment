# Stroke Prediction Using Deep Learning

A state-of-the-art deep learning project for predicting stroke risk based on patient health data.

## Problem Statement

Stroke is a leading cause of death and disability worldwide. Early detection and risk assessment can significantly improve patient outcomes. This project aims to develop a reliable deep learning model for predicting stroke risk based on patient demographic and health data.

## Project Structure

```
stroke-prediction/
├── data/                 # Data storage and processing
│   ├── raw/              # Raw dataset files
│   └── processed/        # Processed dataset files
├── models/               # Saved model checkpoints
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── src/                  # Source code
│   ├── data/             # Data processing utilities
│   ├── models/           # Model architecture definitions
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Helper functions
├── tests/                # Unit tests
├── app/                  # Web application for deployment
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stroke-prediction.git
cd stroke-prediction
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle, which is publicly available and contains the following features:

- Demographic information (age, gender)
- Medical history (hypertension, heart disease)
- Lifestyle factors (smoking status, work type)
- Health metrics (BMI, glucose level)
- Target: Stroke occurrence (0 = No, 1 = Yes)

## Usage

### Data Preprocessing

```bash
python src/data/preprocess.py
```

### Model Training

```bash
python src/training/train.py
```

### Model Evaluation

```bash
python src/evaluation/evaluate.py
```

### Web Application

```bash
python app/app.py
```
Then open your browser and navigate to `http://localhost:5000`

## Model Architecture

The project employs a multi-layer neural network with specialized handling for categorical and numerical features. Techniques include:

- Feature normalization and encoding
- Embedding layers for categorical variables
- Dropout for regularization
- Batch normalization
- Class imbalance handling with weighted loss

## Performance

The model achieves:
- AUC-ROC: ~0.85
- F1-Score: ~0.78
- Precision: ~0.76
- Recall: ~0.81

## Compute Resources

This project is designed to run on free compute resources:
- Training: Google Colab (with free GPU runtime)
- Inference: Local CPU or Colab
- Deployment: Lightweight enough for Heroku free tier

## License

This project is licensed under the MIT License - see the LICENSE file for details. 