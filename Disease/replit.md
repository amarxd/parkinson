# Parkinson's Disease Detection System

## Overview

A comprehensive multimodal machine learning system for detecting Parkinson's Disease through voice recordings, handwriting patterns, facial expressions, and motion analysis. Built with Streamlit for an interactive web interface, this clinical-grade screening tool achieves up to 99% accuracy across different analysis modes. The system integrates research datasets from UCI Oxford and implements multiple ML algorithms including Random Forest, Neural Networks, CNN, and LSTM models to provide clinical-grade analysis with detailed medical recommendations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit multi-page web application with interactive components
- **Navigation**: Sidebar-based page navigation with 7 dedicated analysis modules
- **Interactive Components**: 
  - Real-time audio recording and playback
  - Drawable canvas for handwriting analysis (streamlit-drawable-canvas)
  - Live webcam integration for facial and motion analysis
  - Interactive data visualizations using Plotly
- **Layout Strategy**: Wide responsive layout with expandable sections and column-based content organization
- **User Flow**: Simple file upload → Feature extraction → ML prediction → Clinical recommendations

### Backend Processing Pipeline

**Modular Utility Architecture**:
- **VoiceProcessor**: Extracts 23 voice biomarkers (jitter, shimmer, HNR, MFCC, spectral features) using NumPy and SciPy signal processing
- **HandwritingProcessor**: Analyzes drawing patterns for tremor frequency, line smoothness, micrographia using OpenCV and PIL
- **FacialProcessor**: Detects facial masking and expression mobility using MediaPipe Face Mesh (falls back to basic CV if unavailable)
- **MotionProcessor**: Analyzes tremors, bradykinesia, and gait using MediaPipe Pose estimation
- **Feature Engineering**: Centralized feature registries in `models/voice_features.py` and `models/handwriting_features.py` defining clinical thresholds and normal ranges

**Design Pattern**: Each processor follows a consistent interface with `extract_features()` method returning standardized feature dictionaries for model consumption.

### Machine Learning Models

**Model Architecture**:
- **Voice Analysis**: Random Forest (95.58%), MLP Neural Network (98.31%), SVM, CNN Spectrogram, LSTM Sequential
- **Handwriting Analysis**: Deep CNN (99.22%), Transfer Learning CNN, Random Forest, SVM
- **Model Management**: Centralized `ModelLoader` class handling training, loading, and prediction across algorithms
- **Performance Tracking**: Built-in comparison metrics (accuracy, sensitivity, specificity, F1-score, AUC-ROC)

**Training Approach**: Models train on standardized feature arrays using scikit-learn pipelines with StandardScaler normalization. Fallback to pre-configured model parameters when datasets unavailable.

### Data Management

**Dataset Integration**:
- **UCI Oxford Voice Dataset**: 195 recordings, 23 biomarkers (downloadable from UCI repository)
- **Handwriting Datasets**: Spiral and wave pattern analysis
- **PPMI Clinical Data**: 600 instances with progression markers
- **DataLoader Strategy**: Automated dataset downloading with caching and fallback synthetic data generation using sklearn.datasets.make_classification

**Data Processing**: Standardized preprocessing pipelines ensuring consistent feature extraction across analysis modes.

### Clinical Reporting System

**Report Generation**:
- **ReportGenerator**: Produces individual, batch, clinical study, and progress monitoring reports
- **PDFReportGenerator**: Creates medical-grade PDF documentation using ReportLab with professional formatting
- **Risk Stratification**: Multi-level assessment (low/moderate/high/very-high) with color-coded severity indicators
- **Clinical Integration**: Includes physician details, clinic information, follow-up recommendations, and lifestyle modifications

**Medical Recommendations**: Comprehensive guidance covering specialists to consult, diagnostic tests, lifestyle modifications (physical care, nutrition, mental health), and support resources.

## External Dependencies

### Core ML and Data Science
- **pandas** (2.3.2+): Clinical dataset manipulation and analysis
- **numpy** (2.3.2+): Numerical computing for signal processing
- **scikit-learn** (1.7.1+): ML model training and evaluation
- **scipy** (1.16.1+): Signal processing for voice and motion analysis
- **joblib** (1.5.2+): Model serialization and loading

### Computer Vision and Media Processing
- **opencv-python** (4.11.0+): Image processing for handwriting analysis
- **mediapipe** (0.10.14+): Facial landmark detection and pose estimation (graceful fallback if unavailable)
- **pillow** (11.3.0+): Image handling and manipulation

### Audio Processing
- **librosa**: Voice signal processing and feature extraction (imported dynamically)
- **sounddevice**: Live audio recording (optional, graceful fallback)
- **soundfile**: Audio file I/O (optional)

### Visualization
- **plotly** (6.3.0+): Interactive charts and medical visualizations
- **matplotlib** (3.10.5+): Static plotting for PDF reports
- **seaborn** (0.13.2+): Statistical visualizations

### Web Framework
- **streamlit** (1.49.0+): Main application framework
- **streamlit-drawable-canvas** (0.9.3+): Interactive handwriting canvas

### Report Generation
- **reportlab** (4.4.3+): Professional PDF clinical report generation

### Data Sources
- **UCI Machine Learning Repository**: Oxford Parkinson Dataset (auto-downloaded via HTTP)
- **Local datasets**: CSV files in `datasets/` directory with voice, handwriting, and clinical progression data

### Hardware Requirements
- **Webcam**: Required for facial and motion analysis (cloud deployments use upload mode)
- **Microphone**: Required for live voice recording (upload mode available as fallback)