@echo off
echo Starting League CV Service (Development Mode)
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo ERROR: Virtual environment not found!
    echo Run: python -m venv venv
    echo Then: .\venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if dependencies installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ERROR: Dependencies not installed!
    echo Run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Start server
echo Starting server on http://localhost:8765
echo Press Ctrl+C to stop
echo.
cd src
python main.py
