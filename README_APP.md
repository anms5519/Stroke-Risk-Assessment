# StrokeGuard AI - Stroke Risk Prediction Application

## Overview

StrokeGuard AI is a web-based application that predicts stroke risk based on various health parameters. It uses a combination of medical knowledge and machine learning to provide personalized risk assessments.

## Features

- Modern, responsive user interface
- Interactive risk visualization
- Personalized health recommendations
- Toggle switches for yes/no inputs
- Risk factor importance analysis
- Responsive design for all devices

## Installation Requirements

- Python 3.8 or higher
- Required packages (see requirements.txt)

## Quick Start

### Option 1: Using the Batch File (Recommended)

1. Simply double-click the `StartStrokePredictor.bat` file
2. Follow the on-screen instructions
3. When prompted, open your web browser and navigate to: http://localhost:5000
4. Use the application and close the command window when finished

### Option 2: Manual Start

1. Open a command prompt in the application directory
2. Run: `python run_app.py`
3. Open your web browser and navigate to: http://localhost:5000

## Using the Application

1. Fill out the health information form with your personal data
2. Click "Calculate Risk" to see your results
3. Review your personalized risk assessment and recommendations
4. Print or save your results if desired

## Troubleshooting

- **Application Won't Start**: Make sure Python is installed and in your PATH
- **Browser Connection Error**: Check that the application is running and try http://127.0.0.1:5000
- **Missing Dependencies**: Run `pip install -r requirements.txt` to install required packages

## Data Privacy

All data is processed locally on your machine. No health information is sent to any external servers or stored permanently.

## Disclaimer

This application is for educational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

Developed as a final year machine learning project using Flask, scikit-learn, and Bootstrap. 