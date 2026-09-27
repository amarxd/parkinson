# Parkinson's Disease Detection System - VS Code Setup Guide

## Overview
This is a comprehensive Streamlit-based application for detecting Parkinson's Disease through multimodal analysis including voice, handwriting, facial expressions, and motion detection.

## Quick Start

### 1. Prerequisites
- Python 3.11 or higher
- pip package manager
- Webcam (for facial and motion analysis)
- Microphone (for voice recording)

### 2. Installation Steps

#### Clone or Download the Project
```bash
cd parkinsons-detection-system
```

#### Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Install Dependencies
You can use either method:

**Method 1: Using pyproject.toml (Recommended)**
```bash
pip install -e .
```

**Method 2: Manual installation**
```bash
pip install streamlit pandas numpy plotly scipy scikit-learn opencv-python mediapipe pillow seaborn matplotlib joblib reportlab streamlit-drawable-canvas
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Optional: Audio Recording Features

For real-time audio recording, you need to install PortAudio:

**Windows:**
```bash
pip install sounddevice
```

**Mac:**
```bash
brew install portaudio
pip install sounddevice
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev
pip install sounddevice
```

**Note:** If you encounter audio device issues, the system will gracefully fall back to file upload mode. Audio recording is optional - you can always upload pre-recorded audio files.

## Project Structure

```
.
├── app.py                          # Main application entry point
├── pages/                          # Streamlit pages
│   ├── 1_Voice_Analysis.py         # Voice recording and analysis
│   ├── 2_Handwriting_Analysis.py   # Drawing canvas and handwriting
│   ├── 3_Dataset_Explorer.py       # Dataset visualization
│   ├── 4_Model_Comparison.py       # ML model performance
│   ├── 5_Clinical_Reports.py       # PDF report generation
│   ├── 6_Facial_Analysis.py        # Webcam facial expression
│   └── 7_Motion_Detection.py       # Hand gesture and tremor detection
├── utils/                          # Utility modules
│   ├── voice_processor.py          # Audio feature extraction
│   ├── handwriting_processor.py    # Image processing
│   ├── facial_processor.py         # Face detection and analysis
│   ├── motion_processor.py         # Motion and tremor detection
│   ├── model_loader.py             # ML model management
│   ├── report_generator.py         # PDF report creation
│   └── data_loader.py              # Dataset loading
├── models/                         # Pre-trained ML models
│   ├── voice_*.pkl                 # Voice analysis models
│   └── handwriting_*.pkl           # Handwriting models
├── sample_data/                    # Sample test data
│   ├── audio/                      # Sample voice recordings
│   └── images/                     # Sample handwriting images
├── datasets/                       # Exported datasets (CSV)
│   ├── voice_dataset.csv
│   ├── handwriting_dataset.csv
│   └── ppmi_clinical_dataset.csv
└── pyproject.toml                  # Python dependencies

```

## Features

### 1. Voice Analysis (`1_Voice_Analysis.py`)
- Upload audio files (WAV, MP3, FLAC)
- Real-time voice recording
- Extract features: jitter, shimmer, fundamental frequency
- ML models: Random Forest, MLP, CNN, LSTM

**Sample Audio Files Available:**
- `sample_data/audio/healthy_voice_sample.wav` - Normal voice recording
- `sample_data/audio/parkinsons_voice_sample.wav` - PD voice with high jitter
- `sample_data/audio/parkinsons_tremor_voice.wav` - PD voice with tremor

### 2. Handwriting Analysis (`2_Handwriting_Analysis.py`)
- Interactive drawing canvas
- Upload handwriting images
- Spiral and wave pattern analysis
- Detect tremor and micrographia

**Sample Images Available:**
- `sample_data/images/healthy_spiral.png` - Normal spiral drawing
- `sample_data/images/parkinsons_spiral.png` - PD spiral with tremor
- `sample_data/images/healthy_wave.png` - Normal wave pattern
- `sample_data/images/parkinsons_wave.png` - PD wave with irregularity

### 3. Facial Analysis (`6_Facial_Analysis.py`)
- Upload facial photos for analysis
- Detect facial masking (hypomimia)
- Blinking pattern analysis
- Expression mobility scoring
- **Note:** Real-time webcam analysis is simulated in the current version. For production use, consider integrating `streamlit-webrtc` or similar libraries for live video streaming.

**Requirements:**
- Photos or images for upload analysis
- (Optional) Webcam for future real-time features

### 4. Motion Detection (`7_Motion_Detection.py`)
- Upload videos or images for hand tremor analysis
- Finger tapping test assessment
- Bradykinesia scoring
- Gait pattern analysis from video
- **Note:** Real-time webcam analysis is simulated in the current version. The MediaPipe processors are implemented and can analyze uploaded images/videos.

**Requirements:**
- Images or videos for upload analysis
- (Optional) Webcam for future real-time features

### 5. Clinical Reports (`5_Clinical_Reports.py`)
- Generate comprehensive PDF reports
- Multi-modal analysis integration
- Risk stratification
- Clinical recommendations

### 6. Dataset Explorer (`3_Dataset_Explorer.py`)
- Explore integrated research datasets
- Interactive visualizations
- Statistical analysis

### 7. Model Comparison (`4_Model_Comparison.py`)
- Compare ML model performance
- Accuracy, precision, recall metrics
- ROC curves and confusion matrices

## Available Datasets

All datasets are exported to CSV format in the `datasets/` directory:

### 1. Voice Dataset (`voice_dataset.csv`)
- **Source:** UCI Oxford Parkinson Dataset
- **Samples:** 195 voice recordings
- **Features:** 23 vocal biomarkers (jitter, shimmer, HNR, etc.)
- **Labels:** status (1 = PD, 0 = healthy)

### 2. Handwriting Dataset (`handwriting_dataset.csv`)
- **Samples:** 77 subjects (62 PD patients, 15 controls)
- **Features:** 15 handwriting characteristics
- **Labels:** class (1 = PD, 0 = healthy)

### 3. PPMI Clinical Dataset (`ppmi_clinical_dataset.csv`)
- **Samples:** 600 clinical instances
- **Features:** 15 clinical progression markers
- **Labels:** group (PD or Control)

## Testing the Application

### Quick Test with Sample Data

1. **Test Voice Analysis:**
   - Navigate to "Voice Analysis" page
   - Click "Upload Audio File"
   - Select `sample_data/audio/parkinsons_voice_sample.wav`
   - View analysis results and predictions

2. **Test Handwriting Analysis:**
   - Navigate to "Handwriting Analysis" page
   - Click "Upload Image"
   - Select `sample_data/images/parkinsons_spiral.png`
   - View tremor detection and analysis

3. **Test Facial Analysis:**
   - Navigate to "Facial Analysis" page
   - Click "Upload Photo" mode
   - Upload a facial image for analysis
   - Or click "Start Webcam Analysis" to see simulated metrics demonstration
   - **Note:** For production real-time webcam, additional libraries like `streamlit-webrtc` are needed

4. **Test Motion Detection:**
   - Navigate to "Motion Detection" page
   - Select "Hand Tremor Detection"
   - Click "Start Tremor Detection" to see simulated analysis
   - Or upload hand/motion images for actual MediaPipe processing
   - **Note:** For production real-time webcam, additional libraries are needed

## Troubleshooting

### Import Errors
If you see import errors, ensure all dependencies are installed:
```bash
pip install --upgrade streamlit pandas numpy plotly scipy scikit-learn opencv-python mediapipe pillow seaborn matplotlib joblib reportlab streamlit-drawable-canvas
```

### Webcam & Audio Features
- Real-time video recording works with standard webcams via OpenCV
- Real-time audio recording requires PortAudio (see installation above)
- If camera/microphone access fails, the system gracefully falls back to file upload mode
- All processors (facial, motion, voice) work with both live capture and uploaded files

### Audio Upload Issues
- Ensure audio files are in WAV, MP3, or FLAC format
- Check file is not corrupted
- Try converting to WAV format if issues persist

### Model Loading Errors
- Ensure `models/` directory exists with .pkl files
- Models are pre-trained and should work out of the box

## Performance Notes

- **Webcam features:** Work best with good lighting
- **Voice analysis:** Requires clear audio recordings (minimal background noise)
- **Handwriting:** Works best with high-contrast images (black on white)

## Privacy & Data

- All processing happens locally on your machine
- No data is sent to external servers
- Webcam/microphone access is only used during active analysis
- Sample data is synthetic and for testing purposes only

## Citation

If using this system for research, please cite the original datasets:
- UCI Oxford Parkinson Dataset: Little et al. (2007)
- Handwriting Analysis: Pereira et al. (2016)
- PPMI Dataset: Parkinson's Progression Markers Initiative

## Support

For issues or questions:
1. Check this documentation
2. Review error messages in the terminal
3. Ensure all dependencies are correctly installed
4. Check Python version (3.11+)

## License

This application is for educational and research purposes. The pre-trained models and datasets have their respective licenses.
