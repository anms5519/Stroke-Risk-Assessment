"""
Launcher script for Stroke Prediction Application.
"""

import os
import sys
from app.app import app

if __name__ == "__main__":
    print("Starting Stroke Prediction Application...")
    print("Please wait while the application launches...")
    print("Once started, access the application at: http://localhost:5000")
    
    try:
        # Run the app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down the application...")
        sys.exit(0) 