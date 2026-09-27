import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import wave
import librosa
import librosa.display
import tempfile
import os

from utils.voice_processor import VoiceProcessor
from utils.model_loader import ModelLoader
from utils.data_loader import DataLoader
from utils.live_recording import LiveAudioRecorder
from utils.result_display import display_pd_result, display_medical_disclaimer

# ------------------------------------------------------
# Streamlit Configuration
# ------------------------------------------------------
st.set_page_config(page_title="Voice Analysis", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice Analysis for Parkinson's Detection")
st.markdown("""
Analyze voice recordings using advanced signal processing and machine learning models 
to detect Parkinson's Disease indicators in speech patterns.
""")

# ------------------------------------------------------
# Initialize Processors
# ------------------------------------------------------
@st.cache_resource
def get_processors():
    return VoiceProcessor(), ModelLoader(), DataLoader()


# ------------------------------------------------------
# Load Audio File Function
# ------------------------------------------------------
def _load_audio_file(file_path):
    """Load audio file safely using wave or librosa."""
    # First, try using librosa (it supports mp3, flac, wav, etc.)
    try:
        audio_data, sr = librosa.load(file_path, sr=None)
        return audio_data, sr
    except Exception:
        # If librosa fails, fallback to basic WAV reading
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio_data = audio_data / 32768.0  # Normalize
            sr = wav_file.getframerate()
            return audio_data, sr


# ------------------------------------------------------
# Main Streamlit App
# ------------------------------------------------------
def main():
    voice_processor, model_loader, data_loader = get_processors()

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Analysis Controls")
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Upload Audio File", "Real-time Recording", "Dataset Analysis"]
        )

        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "MLP Neural Network", "CNN Spectrogram", "LSTM Sequential"]
        )

        st.markdown("### 📊 Voice Features")
        show_features = st.multiselect(
            "Display Features",
            ["Jitter", "Shimmer", "Fundamental Frequency", "Harmonics", "Spectral", "All"],
            default=["All"]
        )

    # ------------------------------------------------------
    # Upload Audio File Mode
    # ------------------------------------------------------
    if analysis_mode == "Upload Audio File":
        st.markdown("### 📁 Upload Audio File")
        st.markdown("Supported formats: WAV, MP3, FLAC")

        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac'],
            help="Upload a voice recording for analysis"
        )

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                audio_data, sr = _load_audio_file(tmp_path)

                # Display waveform
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("#### 🔊 Audio Waveform")
                    fig = px.line(
                        x=np.linspace(0, len(audio_data) / sr, len(audio_data)),
                        y=audio_data,
                        title="Audio Waveform",
                        labels={'x': 'Time (seconds)', 'y': 'Amplitude'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.audio(uploaded_file, format='audio/wav')

                with col2:
                    st.markdown("#### 📈 Audio Properties")
                    st.metric("Duration", f"{len(audio_data)/sr:.2f} sec")
                    st.metric("Sample Rate", f"{sr} Hz")
                    st.metric("Channels", "Mono")
                    st.metric("File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")

                # ------------------------------------------------------
                # Feature Extraction
                # ------------------------------------------------------
                if st.button("🔬 Analyze Voice Features", use_container_width=True):
                    with st.spinner("Extracting voice features..."):
                        features = voice_processor.extract_features(audio_data, sr)
                        if features is not None:
                            st.markdown("### 📊 Extracted Voice Features")

                            # Core feature display
                            jitter_shimmer_data = pd.DataFrame({
                                'Feature': ['Jitter RAP', 'Jitter PPQ5', 'Shimmer', 'Shimmer APQ3'],
                                'Value': [
                                    features.get('jitter_rap', 0),
                                    features.get('jitter_ppq5', 0),
                                    features.get('shimmer', 0),
                                    features.get('shimmer_apq3', 0)
                                ]
                            })
                            fig = px.bar(jitter_shimmer_data, x='Feature', y='Value', title="Jitter & Shimmer Analysis")
                            st.plotly_chart(fig, use_container_width=True)

                            # ------------------------------------------------------
                            # Spectrogram Visualization
                            # ------------------------------------------------------
                            st.markdown("### 🎵 Spectrogram Analysis")
                            stft = librosa.stft(audio_data)
                            spectrogram = np.abs(stft)
                            fig = px.imshow(
                                librosa.amplitude_to_db(spectrogram),
                                title="Spectrogram",
                                labels={'x': 'Time', 'y': 'Frequency', 'color': 'dB'},
                                aspect="auto"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # ------------------------------------------------------
                            # MFCC Visualization
                            # ------------------------------------------------------
                            st.markdown("### 🎶 MFCC Features Over Time")
                            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
                            mfcc_df = pd.DataFrame(mfccs.T, columns=[f'MFCC_{i+1}' for i in range(13)])
                            fig = px.line(mfcc_df, title="MFCC Feature Dynamics")
                            st.plotly_chart(fig, use_container_width=True)

                            # ------------------------------------------------------
                            # Parkinson’s Prediction
                            # ------------------------------------------------------
                            model = model_loader.load_voice_model(model_type)
                            prediction, confidence = voice_processor.predict_parkinsons(features, model)

                            st.markdown("### 🎯 Parkinson’s Detection Results")
                            if prediction == 1:
                                st.error(f"⚠️ Parkinson’s Indicators Detected (Confidence: {confidence:.2f}%)")
                            else:
                                st.success(f"✅ No Parkinson’s Indicators Detected (Confidence: {confidence:.2f}%)")

            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
            finally:
                os.unlink(tmp_path)

    # ------------------------------------------------------
    # Real-Time Recording
    # ------------------------------------------------------
    elif analysis_mode == "Real-time Recording":
        st.markdown("### 🎤 Live Audio Recording")
        st.markdown("Record your voice directly using your microphone for real-time analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Recording settings
            recording_duration = st.slider("Recording Duration (seconds)", 3, 15, 5)
            
            st.markdown("""
            **Recording Instructions:**
            1. Ensure your microphone is connected and working
            2. Speak clearly and naturally during the recording
            3. Say sustained vowels (e.g., "Ahhhhh") for at least 3 seconds
            4. Avoid background noise if possible
            """)
            
            if st.button("🎙️ Start Live Recording", use_container_width=True):
                with st.spinner(f"Recording audio for {recording_duration} seconds..."):
                    try:
                        # Initialize audio recorder
                        audio_recorder = LiveAudioRecorder(sample_rate=22050)
                        
                        # Show countdown
                        st.info(f"🔴 Recording in progress... Speak now for {recording_duration} seconds!")
                        
                        # Record audio
                        success, audio_data, sr, message = audio_recorder.record_audio(
                            duration_seconds=recording_duration
                        )
                        
                        if success and audio_data is not None:
                            st.success(message)
                            
                            # Save audio to temp file
                            temp_audio_path = audio_recorder.save_audio(audio_data)
                            
                            # Display audio info
                            audio_info = audio_recorder.get_audio_info(audio_data)
                            
                            st.markdown("#### 🔊 Recording Information")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Duration", f"{audio_info.get('duration', 0):.2f}s")
                            with col_b:
                                st.metric("Sample Rate", f"{audio_info.get('sample_rate', 0)} Hz")
                            with col_c:
                                st.metric("Peak Level", f"{audio_info.get('peak_db', 0):.1f} dB")
                            
                            # Play back recorded audio
                            st.audio(audio_data, sample_rate=sr)
                            
                            # AUTOMATICALLY Analyze the recording
                            with st.spinner("🔬 Analyzing voice features..."):
                                features = voice_processor.extract_features(audio_data, sr)
                                
                                if features is not None:
                                    st.markdown("### 📊 Voice Analysis Results")
                                    
                                    # Display key features
                                    col_x, col_y, col_z = st.columns(3)
                                    with col_x:
                                        jitter = features.get('jitter_rap', 0)
                                        st.metric("Jitter (RAP)", f"{jitter:.5f}",
                                                delta="Normal" if jitter < 0.005 else "Elevated",
                                                delta_color="normal" if jitter < 0.005 else "off")
                                    
                                    with col_y:
                                        shimmer = features.get('shimmer', 0)
                                        st.metric("Shimmer", f"{shimmer:.5f}",
                                                delta="Normal" if shimmer < 0.04 else "Elevated",
                                                delta_color="normal" if shimmer < 0.04 else "off")
                                    
                                    with col_z:
                                        hnr = features.get('hnr', 25)
                                        st.metric("HNR (dB)", f"{hnr:.2f}",
                                                delta="Normal" if hnr > 20 else "Low",
                                                delta_color="normal" if hnr > 20 else "off")
                                    
                                    # Make prediction
                                    model = model_loader.load_voice_model(model_type)
                                    prediction, confidence = voice_processor.predict_parkinsons(features, model)
                                    
                                    # SHOW CLEAR PD RESULT IMMEDIATELY
                                    st.markdown("---")
                                    st.markdown("### 🎯 PARKINSON'S DISEASE DETECTION RESULT")
                                    display_pd_result(prediction, confidence/100 if confidence > 1 else confidence, "Voice Analysis")
                                    display_medical_disclaimer()
                                    
                                    # Store results in session state
                                    st.session_state.voice_analysis_results = {
                                        'features': features,
                                        'prediction': prediction,
                                        'confidence': confidence,
                                        'audio_path': temp_audio_path
                                    }
                                else:
                                    st.error("Could not extract voice features. Please try recording again.")
                        else:
                            st.error(f"Recording failed: {message}")
                            st.info("Please check your microphone permissions and ensure no other application is using it.")
                    
                    except Exception as e:
                        st.error(f"Error during live recording: {str(e)}")
                        st.info("Microphone access may be restricted. Please check system permissions.")
        
        with col2:
            st.markdown("#### 📋 Recording Tips")
            st.markdown("""
            **For Best Results:**
            - Use a good quality microphone
            - Record in a quiet environment
            - Maintain consistent volume
            - Speak naturally
            
            **Recommended Sounds:**
            - Sustained "Ahhhhh" (3-5s)
            - Counting from 1 to 10
            - Reading a short passage
            - Repeating "pa-ta-ka" rapidly
            """)
            
            if 'voice_analysis_results' in st.session_state:
                st.markdown("#### ✅ Latest Results")
                results = st.session_state.voice_analysis_results
                st.metric("Last Prediction", 
                         "PD Detected" if results['prediction'] == 1 else "Normal",
                         f"{results['confidence']:.1f}%")

    # ------------------------------------------------------
    # Dataset Analysis
    # ------------------------------------------------------
    elif analysis_mode == "Dataset Analysis":
        st.markdown("### 📊 Dataset Analysis")
        voice_data = data_loader.load_voice_dataset()
        if voice_data is not None and not voice_data.empty:
            st.dataframe(voice_data.head(10), use_container_width=True)
        else:
            st.warning("No dataset found or empty dataset.")


# ------------------------------------------------------
# Run App
# ------------------------------------------------------
if __name__ == "__main__":
    main()
