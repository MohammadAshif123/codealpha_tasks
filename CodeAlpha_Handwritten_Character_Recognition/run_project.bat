@echo off
echo ================================================
echo CodeAlpha - Handwritten Character Recognition
echo ================================================

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo.
echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Training CNN on MNIST...
python src\train.py

echo.
echo ================================================
echo Training finished.
echo To launch the web app, run:
echo .\.venv\Scripts\python.exe -m streamlit run app.py
echo ================================================

pause
