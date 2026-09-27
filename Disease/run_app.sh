#!/usr/bin/env bash
# Unix helper to activate a virtualenv and run the Streamlit app.
# Edit VENV_PATH to your virtual environment directory.
VENV_PATH="parkinson_env"
if [ -f "$VENV_PATH/bin/activate" ]; then
  source "$VENV_PATH/bin/activate"
else
  echo "Warning: virtual environment activate script not found at $VENV_PATH/bin/activate"
  echo "Continuing with current Python interpreter..."
fi
python -m streamlit run Disease/app.py
