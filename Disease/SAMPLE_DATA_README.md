# Sample Data Guide

## Overview
This directory contains sample data files for testing the Parkinson's Disease Detection System. All samples are synthetic but designed to mimic real clinical characteristics.

## Directory Structure

```
sample_data/
├── audio/              # Voice recording samples
└── images/             # Handwriting sample images
```

## Audio Samples

### 1. healthy_voice_sample.wav
- **Type:** Healthy control voice
- **Characteristics:**
  - Low jitter (< 0.5%)
  - Low shimmer (< 2.5%)
  - Stable fundamental frequency (~150 Hz)
  - Good harmonics-to-noise ratio
- **Use Case:** Baseline comparison for voice analysis

### 2. parkinsons_voice_sample.wav
- **Type:** Parkinson's Disease voice
- **Characteristics:**
  - High jitter (> 0.8%)
  - High shimmer (> 4%)
  - Variable fundamental frequency
  - Reduced harmonics-to-noise ratio
- **Use Case:** Testing PD detection algorithms

### 3. parkinsons_tremor_voice.wav
- **Type:** Parkinson's voice with vocal tremor
- **Characteristics:**
  - Vocal tremor at ~5 Hz (typical PD frequency)
  - Amplitude modulation
  - Increased noise levels
- **Use Case:** Tremor-specific voice analysis

## How to Use Audio Samples

1. **In the application:**
   - Navigate to "Voice Analysis" page
   - Click "Upload Audio File"
   - Select any of the audio samples
   - Review the extracted features and predictions

2. **Expected Results:**
   - `healthy_voice_sample.wav` should predict "Healthy" or low PD probability
   - `parkinsons_voice_sample.wav` should predict "Parkinson's" or high PD probability
   - `parkinsons_tremor_voice.wav` should detect tremor characteristics

## Image Samples

### 1. healthy_spiral.png
- **Type:** Normal spiral drawing
- **Characteristics:**
  - Smooth, continuous lines
  - Regular spacing between loops
  - Minimal tremor
  - Good size consistency
- **Use Case:** Baseline for handwriting analysis

### 2. parkinsons_spiral.png
- **Type:** PD spiral with tremor
- **Characteristics:**
  - Irregular, jagged lines
  - Tremor oscillations visible
  - Micrographia (smaller size)
  - Reduced smoothness
- **Use Case:** Testing tremor and micrographia detection

### 3. healthy_wave.png
- **Type:** Normal wave pattern
- **Characteristics:**
  - Consistent amplitude
  - Regular frequency
  - Smooth curves
  - Good spatial control
- **Use Case:** Wave pattern analysis

### 4. parkinsons_wave.png
- **Type:** PD wave with irregularities
- **Characteristics:**
  - Variable amplitude
  - Tremor patterns visible
  - Less smooth curves
  - Reduced consistency
- **Use Case:** Testing motor dysfunction detection

## How to Use Image Samples

1. **In the application:**
   - Navigate to "Handwriting Analysis" page
   - Select "Upload Image" mode
   - Upload any of the sample images
   - Review tremor detection and motor analysis

2. **Expected Results:**
   - Healthy samples should show low tremor scores and good motor control
   - PD samples should detect tremor, micrographia, and motor dysfunction

## Testing Webcam Features

For facial and motion detection, use the real-time webcam features:

1. **Facial Analysis:**
   - Good lighting is essential
   - Face the camera directly
   - Follow expression test prompts
   - System will analyze facial mobility and blinking patterns

2. **Motion Detection:**
   - Position hands clearly in camera view
   - Perform finger tapping tests
   - Hold hands steady for tremor detection
   - Follow gait analysis instructions if testing full body

## Creating Your Own Test Data

### Audio Files
To create your own test audio:
1. Record voice saying "ahhhhh" for 3-5 seconds
2. Save as WAV format
3. Upload to the Voice Analysis page

### Handwriting Images
To create your own test images:
1. Use the interactive canvas in "Handwriting Analysis"
2. Draw spirals or waves as instructed
3. System will analyze in real-time
4. Alternatively, scan/photo physical drawings (high contrast, black on white)

## Technical Specifications

### Audio Requirements
- **Format:** WAV, MP3, or FLAC
- **Sample Rate:** 16-48 kHz (22050 Hz recommended)
- **Channels:** Mono or stereo
- **Duration:** 2-10 seconds
- **Quality:** Clear audio, minimal background noise

### Image Requirements
- **Format:** PNG, JPG, or JPEG
- **Resolution:** 300x300 pixels minimum
- **Quality:** High contrast (black drawing on white background)
- **Content:** Clear, complete drawings

## Notes

- All sample files are synthetic and created for demonstration purposes
- Actual clinical data would show more complex patterns
- Results may vary based on system configuration
- Sample data is not suitable for clinical diagnosis

## Privacy

- All sample data is synthetic
- No real patient data is included
- Feel free to use for testing and development
