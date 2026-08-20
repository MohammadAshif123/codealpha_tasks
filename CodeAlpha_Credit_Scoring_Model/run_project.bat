@echo off
echo ============================================
echo CodeAlpha Credit Scoring Model
echo ============================================

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate

echo Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Training models...
python src\train.py

echo.
echo Running sample prediction...
python src\predict.py

echo.
echo Done.
pause
