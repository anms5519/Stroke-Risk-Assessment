@echo off
title StrokeGuard AI - Stroke Risk Prediction Tool
color 0B
cls

echo ========================================================================
echo                        StrokeGuard AI - v1.0.0
echo                   Stroke Risk Prediction Application
echo ========================================================================
echo.
echo  This application will help predict the risk of stroke based on health data.
echo  It uses machine learning to analyze various health factors and provide
echo  personalized risk assessment.
echo.
echo  When the application starts, please open your web browser and go to:
echo  http://localhost:5000
echo.
echo  Press any key to start the application...
echo  Press Ctrl+C to exit the application when done.
echo.
echo ========================================================================
pause > nul

cls
echo ========================================================================
echo                        STARTING APPLICATION
echo ========================================================================
echo.
echo  [*] Initializing application components...
timeout /t 2 > nul
echo  [*] Starting web server...
echo  [*] Loading prediction models...
echo.
echo  Application is starting. Please wait...
echo.
echo ========================================================================
echo.

python run_app.py

:end
echo.
echo ========================================================================
echo                        APPLICATION CLOSED
echo ========================================================================
echo.
echo  Thank you for using StrokeGuard AI.
echo.
echo  Press any key to exit...
pause > nul 