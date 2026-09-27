@echo off
REM Windows helper to activate a virtualenv and run the Streamlit app.
REM Edit VENV_PATH to the path of your virtual environment (relative or absolute).
set VENV_PATH=parkinson_env
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
) else (
    echo Warning: virtual environment activate script not found at %VENV_PATH%\Scripts\activate.bat
    echo Continuing with current Python interpreter...
)
python -m streamlit run Disease/app.py
pause
