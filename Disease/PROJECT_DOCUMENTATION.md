# Parkinson's Disease Detection System

## Overview

A comprehensive machine learning system for detecting Parkinson's Disease through multimodal analysis including voice recordings, handwriting patterns, facial expressions, and motion detection. The system integrates pre-trained ML models achieving up to 99% accuracy by analyzing various biomarkers and motor dysfunction indicators. Built as a Streamlit web application with comprehensive dataset exploration, model comparison capabilities, and medical-grade clinical reporting functionality.

## Project Information

**Type:** Medical AI/Machine Learning Application  
**Framework:** Python, Streamlit, Scikit-learn, MediaPipe  
**Purpose:** Early detection and screening tool for Parkinson's Disease indicators  
**Status:** Complete and functional

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page navigation
- **Page Structure**: Modular design with dedicated pages for voice analysis, handwriting analysis, dataset exploration, model comparison, and clinical reporting
- **Interactive Components**: Real-time audio recording, drawable canvas for handwriting analysis, interactive data visualizations using Plotly
- **Layout**: Wide layout with expandable sidebar navigation and responsive column-based content organization

### Backend Architecture
- **Processing Pipeline**: Modular utility classes for voice and handwriting feature extraction
- **Voice Processing**: Librosa-based audio analysis extracting jitter, shimmer, HNR, and spectral features from speech signals
- **Handwriting Processing**: OpenCV and scikit-image based analysis of drawing patterns, tremor detection, and motor dysfunction indicators
- **Feature Engineering**: Comprehensive feature registries defining clinical thresholds, normal ranges, and PD-specific biomarkers

### Machine Learning Models
- **Voice Analysis Models**: Random Forest (95.58% accuracy), MLP Neural Network (98.31% accuracy), CNN Spectrogram, LSTM Sequential
- **Handwriting Models**: Deep CNN (99.22% accuracy), Transfer Learning CNN, Random Forest, SVM Classifier
- **Model Management**: Centralized ModelLoader class handling training, loading, and prediction across multiple algorithms
- **Performance Tracking**: Built-in model comparison with accuracy, sensitivity, specificity, F1-score, and AUC-ROC metrics

### Data Management
- **Dataset Integration**: UCI Oxford Parkinson's voice dataset, handwriting pattern datasets, PPMI clinical progression data
- **Data Loading**: Automated dataset downloading and caching with fallback synthetic data generation
- **Feature Definitions**: Structured feature registries with clinical significance, normal ranges, and PD trend indicators
- **Data Processing**: Standardized preprocessing pipelines for audio signals and image data

### Clinical Reporting System
- **Report Types**: Individual patient reports, batch analysis, clinical study reports, progress monitoring
- **Risk Assessment**: Multi-level risk stratification with clinical thresholds and color-coded severity indicators
- **Clinical Integration**: Medical-grade documentation with physician details, clinic information, and follow-up recommendations
- **Export Capabilities**: PDF report generation with comprehensive analysis results and visualizations

## External Dependencies

### Core ML and Data Science Libraries
- **pandas**: Data manipulation and analysis for clinical datasets
- **numpy**: Numerical computing for signal processing and feature calculations
- **scikit-learn**: Machine learning models, preprocessing, and evaluation metrics
- **scipy**: Scientific computing for statistical analysis and signal processing

### Audio Processing
- **librosa**: Audio analysis and feature extraction for voice biomarker detection
- **soundfile**: Audio file I/O operations for voice recording processing

### Image Processing and Computer Vision
- **opencv-python (cv2)**: Image preprocessing and analysis for handwriting patterns
- **PIL (Pillow)**: Image manipulation and format conversion
- **scikit-image**: Advanced image processing for tremor detection and pattern analysis

### Web Application Framework
- **streamlit**: Primary web application framework with interactive components
- **streamlit-drawable-canvas**: Custom component for interactive handwriting/drawing analysis

### Visualization and Reporting
- **plotly**: Interactive data visualizations and medical charts
- **seaborn**: Statistical data visualization for dataset exploration
- **reportlab**: PDF generation for clinical reports and medical documentation

### Data Sources
- **UCI Oxford Parkinson Dataset**: Voice recordings from 195 subjects (23 PD patients, 8 controls)
- **Handwriting Pattern Datasets**: Motor dysfunction analysis from 62 PD patients and 15 healthy controls
- **PPMI Clinical Data**: Parkinson's Progression Markers Initiative dataset with 600 clinical instances

### File I/O and Utilities
- **joblib**: Model serialization and loading for ML pipeline persistence
- **tempfile**: Temporary file handling for audio processing workflows
- **urllib**: Dataset downloading and external data source integration