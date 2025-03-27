"""
Setup script for creating a Windows executable.
"""

import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {
    "packages": [
        "flask", 
        "numpy", 
        "pandas", 
        "sklearn", 
        "joblib", 
        "os", 
        "sys",
        "pickle",
    ],
    "include_files": [
        ("app/templates/", "app/templates/"),
        ("app/static/", "app/static/"),
        ("src/", "src/"),
    ],
}

# Base for GUI or Console application
base = None
if sys.platform == "win32":
    base = "Console"  # use "Win32GUI" for a GUI app

setup(
    name="StrokeRiskPredictor",
    version="1.0.0",
    description="Stroke Risk Prediction Application",
    options={"build_exe": build_exe_options},
    executables=[Executable("run_app.py", base=base, target_name="StrokeRiskPredictor.exe")]
) 