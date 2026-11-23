@echo off
setlocal
echo ==========================================
echo OULAD Paper 2 - 50%% Boundary Pipeline
echo ==========================================
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
echo.
echo Pipeline finished. Check results\tables and results\figures.
pause
