# Dependencies for Parkinson's Disease Detection System

## Required Python Packages

Your project already includes these core dependencies in `pyproject.toml`:

### Core Framework
- pip install streamlit>=1.49.0
- pip install streamlit-drawable-canvas>=0.9.3

### Data Science & Machine Learning
- pip install numpy>=2.3.2
- pip install pandas>=2.3.2
- pip install scikit-learn>=1.7.1
- pip install scipy>=1.16.1
- pip install joblib>=1.5.2

### Computer Vision & Image Processing
- pip install opencv-python>=4.11.0.86
- pip install pillow>=11.3.0
/* py -3.10 -m venv my_env
    .\my_env\Scripts\activate
    python --version */
- pip install mediapipe>=0.10.14

### Visualization & Plotting
- pip install plotly>=6.3.0
- pip install matplotlib>=3.10.5
- pip install seaborn>=0.13.2

### Report Generation
- pip install reportlab>=4.4.3

## Installation Instructions



### For Local Development (VS Code, PyCharm, etc.):

1. **Create virtual environment:**
   ```bash
   python -m venv parkinson_env
   source parkinson_env/bin/activate  # On Windows: parkinson_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit streamlit-drawable-canvas
   pip install numpy pandas scikit-learn scipy joblib
   pip install opencv-python pillow mediapipe
   pip install plotly matplotlib seaborn
   pip install reportlab librosa soundfile
   ```

3. **For audio recording features (optional):**
   
   **Windows:**
   ```bash
   pip install sounddevice
   ```
   
   **Mac:**
   ```bash
   brew install portaudio
   pip install sounddevice
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install portaudio19-dev
   pip install sounddevice
   ```
   
   Note: If audio recording doesn't work, you can still upload audio files for analysis.

4. **Run the application:**
   ```bash
   # Recommended: use the Python module so the correct interpreter is used
   python -m streamlit run Disease/app.py
   ```

## System Features

✅ **Voice Analysis** - Hypophonia, dysarthria, monotone speech detection  
✅ **Handwriting Analysis** - Micrographia and tremor detection  
✅ **Facial Analysis** - Facial masking and expression mobility assessment  
✅ **Motion Detection** - Tremors, bradykinesia, and gait analysis  
✅ **Clinical Reports** - Comprehensive medical documentation  
✅ **Real-time Analysis** - Live webcam and interactive testing  

## Browser Requirements

- Modern web browser (Chrome, Firefox, Safari, Edge)
- Camera access for facial and motion analysis
- Microphone access for voice analysis

## Running the Application

The application runs on port 5000 by default (Streamlit default). Recommended command:
```bash
# From the repository root:
python -m streamlit run Disease/app.py
streamlit run app.py --server.port 8000
```

Access the application at: http://localhost:8000