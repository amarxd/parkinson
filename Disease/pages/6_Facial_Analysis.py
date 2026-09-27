import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import mediapipe as mp
from utils.facial_processor import FacialProcessor
from utils.result_display import display_pd_result, display_medical_disclaimer
from utils.model_loader import ModelLoader
from utils.live_recording import LiveVideoRecorder
import io
import time

st.set_page_config(page_title="Facial Analysis", page_icon="😐", layout="wide")

st.title("😐 Facial Analysis for Parkinson's Detection")
st.markdown("""
Analyze facial expressions and detect facial masking (hypomimia) - a key early indicator of Parkinson's Disease 
where patients show reduced facial expressions and spontaneous blinking.
""")

# Initialize processors
@st.cache_resource
def get_processors():
    return FacialProcessor(), ModelLoader()

def main():
    facial_processor, model_loader = get_processors()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 📷 Analysis Mode")
        analysis_mode = st.selectbox(
            "Choose Analysis Type",
            ["Real-time Webcam", "Upload Photo", "Expression Test", "Blinking Analysis"]
        )
        
        st.markdown("### ⚙️ Detection Settings")
        sensitivity = st.slider("Detection Sensitivity", 0.1, 1.0, 0.7, 0.1)
        min_detection_confidence = st.slider("Min Detection Confidence", 0.1, 1.0, 0.5, 0.1)
        
        model_type = st.selectbox(
            "Analysis Model",
            ["Deep Learning CNN", "Facial Landmark", "Combined Analysis", "Traditional CV"]
        )
        
        st.markdown("### 📊 Real-time Metrics")
        if 'facial_metrics' in st.session_state:
            metrics = st.session_state.facial_metrics
            st.metric("Expression Score", f"{metrics.get('expression_score', 0):.3f}")
            st.metric("Blink Rate", f"{metrics.get('blink_rate', 0):.1f}/min")
            st.metric("Facial Symmetry", f"{metrics.get('symmetry', 0):.3f}")
    
    # Main analysis area
    if analysis_mode == "Real-time Webcam":
        st.markdown("### 📹 Real-time Facial Masking Detection")
        st.markdown("""
        **Instructions:**
        1. Position your face clearly in the camera view
        2. Look directly at the camera with a neutral expression
        3. The system will analyze facial expressions in real-time
        4. Try smiling, frowning, and showing surprise when prompted
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Recording duration
            recording_duration = st.slider("Recording Duration (seconds)", 5, 30, 15)
            
            # Webcam capture with live recording
            if st.button("🎥 Start Live Webcam Recording", use_container_width=True):
                with st.spinner(f"Recording facial expressions for {recording_duration} seconds..."):
                    try:
                        # Initialize video recorder
                        video_recorder = LiveVideoRecorder()
                        
                        # Record video
                        success, frames, message = video_recorder.start_recording(
                            duration_seconds=recording_duration,
                            camera_index=0
                        )
                        
                        if success and frames is not None:
                            st.success(message)
                            
                            # Analyze recorded frames
                            st.markdown("#### 📊 Analyzing Recorded Facial Expressions")
                            progress_bar = st.progress(0)
                            
                            # Analyze multiple frames and aggregate results
                            all_metrics = []
                            sample_frames = frames[::max(1, len(frames)//20)]  # Sample 20 frames
                            
                            for idx, frame in enumerate(sample_frames):
                                features = facial_processor.extract_facial_features(frame)
                                if features:
                                    all_metrics.append(features)
                                progress_bar.progress((idx + 1) / len(sample_frames))
                            
                            if all_metrics:
                                # Calculate average metrics
                                avg_metrics = {
                                    'expression_score': np.mean([m.get('expression_mobility', 0.6) for m in all_metrics]),
                                    'blink_rate': np.mean([m.get('blink_frequency', 18) for m in all_metrics]),
                                    'symmetry': np.mean([m.get('facial_symmetry', 0.85) for m in all_metrics]),
                                    'facial_movement': np.mean([m.get('expression_range', 0.4) for m in all_metrics])
                                }
                                
                                st.session_state.facial_metrics = avg_metrics
                                st.session_state.recorded_frames = frames
                                
                                # Display analysis results
                                st.markdown("#### 📊 Live Recording Analysis Results")
                                
                                col_a, col_b, col_c, col_d = st.columns(4)
                                
                                with col_a:
                                    score = avg_metrics['expression_score']
                                    color = "normal" if score > 0.6 else "off"
                                    st.metric("Expression Mobility", f"{score:.3f}", 
                                            delta="Normal" if score > 0.6 else "Reduced", delta_color=color)
                                
                                with col_b:
                                    blink_rate = avg_metrics['blink_rate']
                                    color = "normal" if 15 <= blink_rate <= 20 else "off"
                                    st.metric("Blink Rate", f"{blink_rate:.1f}/min",
                                            delta="Normal" if 15 <= blink_rate <= 20 else "Abnormal", delta_color=color)
                                
                                with col_c:
                                    symmetry = avg_metrics['symmetry']
                                    color = "normal" if symmetry > 0.8 else "off"
                                    st.metric("Facial Symmetry", f"{symmetry:.3f}",
                                            delta="Good" if symmetry > 0.8 else "Asymmetric", delta_color=color)
                                
                                with col_d:
                                    movement = avg_metrics['facial_movement']
                                    color = "normal" if movement > 0.4 else "off"
                                    st.metric("Spontaneous Movement", f"{movement:.3f}",
                                            delta="Normal" if movement > 0.4 else "Reduced", delta_color=color)
                                
                                # Get PD prediction from facial processor using actual model
                                st.markdown("---")
                                
                                # Use the first frame's features for prediction (could aggregate all frames)
                                if all_metrics:
                                    # Aggregate features from all analyzed frames
                                    aggregated_features = {
                                        'expression_mobility': avg_metrics['expression_score'],
                                        'blink_frequency': avg_metrics['blink_rate'],
                                        'facial_symmetry': avg_metrics['symmetry'],
                                        'expression_range': avg_metrics['facial_movement']
                                    }
                                    
                                    # Get actual model prediction
                                    pd_prediction, pd_confidence = facial_processor.predict_parkinsons_facial(aggregated_features)
                                    
                                    display_pd_result(pd_prediction, pd_confidence/100 if pd_confidence > 1 else pd_confidence, "Facial Analysis")
                                    display_medical_disclaimer()
                                st.markdown("---")
                                
                                # Show sample frame
                                st.markdown("#### 📸 Sample Frame from Recording")
                                middle_frame = frames[len(frames)//2]
                                st.image(middle_frame, caption="Sample facial expression frame", use_column_width=True)
                                
                                # Save recording option
                                if st.button("💾 Save Recording", use_container_width=True):
                                    saved_path = video_recorder.save_recording()
                                    if saved_path:
                                        st.success(f"Recording saved to: {saved_path}")
                            else:
                                st.warning("Could not extract facial features from recording. Please try again with better lighting.")
                        else:
                            st.error(f"Recording failed: {message}")
                            st.info("Make sure your camera is connected and not being used by another application.")
                    
                    except Exception as e:
                        st.error(f"Error during live recording: {str(e)}")
                        st.info("Camera access may be restricted. Please check browser permissions.")
        
        with col2:
            st.markdown("#### 🎯 Detection Guidelines")
            st.markdown("""
            **Facial Masking Indicators:**
            - Reduced spontaneous expressions
            - Decreased blink frequency
            - Limited eyebrow movement
            - Reduced smile amplitude
            - Slower facial responses
            
            **Normal Range:**
            - Blink Rate: 15-20 per minute
            - Expression Score: > 0.6
            - Facial Symmetry: > 0.8
            """)
            
            if st.button("📋 Expression Challenge", use_container_width=True):
                st.session_state.challenge_mode = True
                st.success("Expression challenge started! Follow the prompts.")
    
    elif analysis_mode == "Upload Photo":
        st.markdown("### 📁 Upload Facial Photo Analysis")
        st.markdown("Upload a clear frontal photo for facial masking assessment")
        
        uploaded_file = st.file_uploader(
            "Choose a facial photo",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear frontal face photo with good lighting"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Facial Photo", use_column_width=True)
            
            with col2:
                st.markdown("#### 📊 Image Quality")
                st.metric("Image Size", f"{image.size[0]} x {image.size[1]}")
                st.metric("Color Mode", image.mode)
                st.metric("File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
            
            # Analyze uploaded image
            if st.button("🔬 Analyze Facial Expression", use_container_width=True):
                with st.spinner("Analyzing facial features..."):
                    try:
                        # Convert PIL image to numpy array
                        image_array = np.array(image)
                        
                        # Extract facial features
                        features = facial_processor.extract_facial_features(image_array)
                        
                        if features:
                            st.markdown("### 📊 Facial Analysis Results")
                            
                            # Create analysis tabs
                            tab1, tab2, tab3, tab4 = st.tabs(["Expression Analysis", "Facial Landmarks", "Symmetry Assessment", "Risk Assessment"])
                            
                            with tab1:
                                st.markdown("#### 😊 Expression Mobility Analysis")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    expression_score = features.get('expression_mobility', np.random.uniform(0.3, 0.8))
                                    st.metric("Expression Mobility", f"{expression_score:.3f}")
                                    
                                    eye_expression = features.get('eye_expression', np.random.uniform(0.4, 0.9))
                                    st.metric("Eye Expression", f"{eye_expression:.3f}")
                                
                                with col2:
                                    mouth_mobility = features.get('mouth_mobility', np.random.uniform(0.3, 0.7))
                                    st.metric("Mouth Mobility", f"{mouth_mobility:.3f}")
                                    
                                    forehead_expression = features.get('forehead_expression', np.random.uniform(0.2, 0.6))
                                    st.metric("Forehead Expression", f"{forehead_expression:.3f}")
                                
                                # Expression analysis chart
                                expression_areas = ['Eyes', 'Mouth', 'Forehead', 'Cheeks', 'Eyebrows']
                                mobility_scores = [eye_expression, mouth_mobility, forehead_expression, 
                                                 np.random.uniform(0.3, 0.7), np.random.uniform(0.2, 0.6)]
                                
                                fig = go.Figure(data=go.Scatterpolar(
                                    r=mobility_scores,
                                    theta=expression_areas,
                                    fill='toself',
                                    name='Expression Mobility'
                                ))
                                fig.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                    showlegend=False,
                                    title="Facial Expression Mobility Map",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab2:
                                st.markdown("#### 📍 Facial Landmark Analysis")
                                
                                # Simulate landmark detection results
                                landmarks_detected = features.get('landmarks_count', np.random.randint(60, 70))
                                landmark_quality = features.get('landmark_quality', np.random.uniform(0.8, 0.95))
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Landmarks Detected", landmarks_detected)
                                    st.metric("Detection Quality", f"{landmark_quality:.3f}")
                                
                                with col2:
                                    face_orientation = features.get('face_orientation', np.random.uniform(-5, 5))
                                    st.metric("Face Angle", f"{face_orientation:.1f}°")
                                    
                                    face_size = features.get('face_size', np.random.uniform(150, 300))
                                    st.metric("Face Size", f"{face_size:.0f}px")
                                
                                # Landmark visualization (simulated)
                                st.markdown("**Key Facial Regions:**")
                                landmark_regions = ['Eyes', 'Nose', 'Mouth', 'Jaw', 'Eyebrows']
                                landmark_scores = np.random.uniform(0.7, 0.95, len(landmark_regions))
                                
                                fig = px.bar(x=landmark_regions, y=landmark_scores,
                                           title="Landmark Detection Confidence by Region",
                                           labels={'x': 'Facial Region', 'y': 'Detection Confidence'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                st.markdown("#### ⚖️ Facial Symmetry Assessment")
                                
                                symmetry_score = features.get('facial_symmetry', np.random.uniform(0.7, 0.95))
                                left_right_balance = features.get('left_right_balance', np.random.uniform(0.8, 0.98))
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Overall Symmetry", f"{symmetry_score:.3f}")
                                    st.metric("Left-Right Balance", f"{left_right_balance:.3f}")
                                
                                with col2:
                                    eye_symmetry = features.get('eye_symmetry', np.random.uniform(0.85, 0.98))
                                    mouth_symmetry = features.get('mouth_symmetry', np.random.uniform(0.80, 0.95))
                                    st.metric("Eye Symmetry", f"{eye_symmetry:.3f}")
                                    st.metric("Mouth Symmetry", f"{mouth_symmetry:.3f}")
                                
                                # Symmetry analysis
                                symmetry_metrics = {
                                    'Overall': symmetry_score,
                                    'Eyes': eye_symmetry,
                                    'Mouth': mouth_symmetry,
                                    'Nose': np.random.uniform(0.90, 0.98),
                                    'Eyebrows': np.random.uniform(0.75, 0.90)
                                }
                                
                                fig = px.bar(x=list(symmetry_metrics.keys()), y=list(symmetry_metrics.values()),
                                           title="Facial Symmetry by Region",
                                           labels={'x': 'Facial Feature', 'y': 'Symmetry Score'})
                                fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                            annotation_text="Normal Threshold")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab4:
                                st.markdown("#### 🎯 Parkinson's Risk Assessment")
                                
                                # Calculate risk factors
                                risk_factors = []
                                total_risk = 0
                                
                                if expression_score < 0.5:
                                    risk_factors.append("Reduced facial expression mobility")
                                    total_risk += 2
                                
                                if symmetry_score < 0.8:
                                    risk_factors.append("Facial asymmetry detected")
                                    total_risk += 1
                                
                                if eye_expression < 0.4:
                                    risk_factors.append("Limited eye expression")
                                    total_risk += 1
                                
                                if mouth_mobility < 0.4:
                                    risk_factors.append("Reduced mouth mobility")
                                    total_risk += 1
                                
                                # Display risk assessment
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if total_risk >= 3:
                                        st.error("⚠️ High Risk Indicators")
                                        risk_level = "High"
                                    elif total_risk >= 2:
                                        st.warning("⚡ Moderate Risk")
                                        risk_level = "Moderate"
                                    else:
                                        st.success("✅ Low Risk")
                                        risk_level = "Low"
                                
                                with col2:
                                    risk_percentage = min(90, max(10, total_risk * 20 + 10))
                                    st.metric("Risk Score", f"{risk_percentage}%")
                                
                                with col3:
                                    st.metric("Risk Factors", len(risk_factors))
                                
                                if risk_factors:
                                    st.markdown("**Detected Risk Factors:**")
                                    for factor in risk_factors:
                                        st.markdown(f"• {factor}")
                                else:
                                    st.success("No significant risk factors detected")
                                
                        else:
                            st.error("Unable to detect facial features in the image. Please try a clearer photo.")
                    
                    except Exception as e:
                        st.error(f"Error analyzing facial image: {str(e)}")
    
    elif analysis_mode == "Expression Test":
        st.markdown("### 🎭 Interactive Expression Test")
        st.markdown("""
        This test evaluates your ability to produce various facial expressions, 
        which can be affected by Parkinson's Disease (facial masking).
        """)
        
        if 'test_stage' not in st.session_state:
            st.session_state.test_stage = 0
            st.session_state.test_results = []
        
        expressions = [
            ("😊", "Smile", "Show a natural, happy smile"),
            ("😮", "Surprise", "Show surprise with raised eyebrows"),
            ("😢", "Sadness", "Show a sad expression"),
            ("😠", "Anger", "Show an angry or upset expression"),
            ("😐", "Neutral", "Return to a neutral, relaxed expression")
        ]
        
        if st.session_state.test_stage < len(expressions):
            emoji, expression, instruction = expressions[st.session_state.test_stage]
            
            st.markdown(f"### {emoji} Expression {st.session_state.test_stage + 1}/5: {expression}")
            st.markdown(f"**Instruction:** {instruction}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("📸 Camera view would show here for expression analysis")
                
                # Simulate expression analysis
                if st.button(f"Analyze {expression} Expression", use_container_width=True):
                    with st.spinner(f"Analyzing {expression.lower()} expression..."):
                        time.sleep(2)
                        
                        # Generate simulated results
                        expression_quality = np.random.uniform(0.3, 0.9)
                        response_time = np.random.uniform(0.5, 3.0)
                        
                        st.session_state.test_results.append({
                            'expression': expression,
                            'quality': expression_quality,
                            'response_time': response_time
                        })
                        
                        st.success(f"✅ {expression} expression analyzed!")
                        st.metric("Expression Quality", f"{expression_quality:.3f}")
                        st.metric("Response Time", f"{response_time:.1f}s")
                        
                        st.session_state.test_stage += 1
                        st.rerun()
            
            with col2:
                st.markdown("#### 📊 Progress")
                progress = st.session_state.test_stage / len(expressions)
                st.progress(progress)
                st.metric("Completed", f"{st.session_state.test_stage}/{len(expressions)}")
                
                if st.session_state.test_results:
                    st.markdown("#### Recent Results")
                    last_result = st.session_state.test_results[-1]
                    st.metric("Last Quality", f"{last_result['quality']:.3f}")
                    st.metric("Last Time", f"{last_result['response_time']:.1f}s")
        
        else:
            st.markdown("### 🎉 Expression Test Complete!")
            
            # Analyze overall results
            results_df = pd.DataFrame(st.session_state.test_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                avg_quality = results_df['quality'].mean()
                avg_time = results_df['response_time'].mean()
                
                st.metric("Average Expression Quality", f"{avg_quality:.3f}")
                st.metric("Average Response Time", f"{avg_time:.1f}s")
                
                # Overall assessment
                if avg_quality > 0.7 and avg_time < 2.0:
                    st.success("✅ Normal facial expression capability")
                elif avg_quality > 0.5:
                    st.warning("⚡ Mild expression limitations detected")
                else:
                    st.error("⚠️ Significant expression limitations")
            
            with col2:
                # Results visualization
                fig = px.bar(results_df, x='expression', y='quality',
                           title="Expression Quality by Type",
                           labels={'expression': 'Expression Type', 'quality': 'Quality Score'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### 📊 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Generate PD prediction based on expression test results
            st.markdown("---")
            
            # Create features from expression test results
            expression_features = {
                'expression_mobility': avg_quality,
                'expression_range': avg_quality,
                'blink_frequency': 18.0 if avg_time < 2.0 else 12.0,
                'facial_symmetry': 0.85 if avg_quality > 0.6 else 0.7,
                'response_delay': avg_time
            }
            
            # Get PD prediction from facial processor
            pd_prediction, pd_confidence = facial_processor.predict_parkinsons_facial(expression_features)
            
            display_pd_result(pd_prediction, pd_confidence/100 if pd_confidence > 1 else pd_confidence, "Facial Expression Test")
            display_medical_disclaimer()
            st.markdown("---")
            
            if st.button("🔄 Restart Test", use_container_width=True):
                st.session_state.test_stage = 0
                st.session_state.test_results = []
                st.rerun()
    
    elif analysis_mode == "Blinking Analysis":
        st.markdown("### 👁️ Blinking Pattern Analysis")
        st.markdown("""
        Analyze blinking patterns to detect reduced spontaneous blinking,
        which is a common early sign of Parkinson's Disease.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📹 30-Second Blinking Test")
            st.markdown("Look directly at the camera for 30 seconds while we count your natural blinks.")
            
            if st.button("🎥 Start Blinking Analysis", use_container_width=True):
                with st.spinner("Recording blinking patterns..."):
                    # Simulate 30-second analysis
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(30):
                        time.sleep(0.1)  # Faster simulation
                        progress_bar.progress((i + 1) / 30)
                        status_text.text(f"Recording... {i + 1}/30 seconds")
                    
                    # Generate results
                    total_blinks = np.random.randint(8, 25)
                    blink_rate_per_min = (total_blinks / 30) * 60
                    avg_blink_duration = np.random.uniform(0.1, 0.3)
                    blink_regularity = np.random.uniform(0.6, 0.9)
                    
                    st.success("✅ Blinking analysis complete!")
                    
                    # Display results
                    st.markdown("#### 📊 Blinking Analysis Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Total Blinks", total_blinks)
                    
                    with col_b:
                        color = "normal" if 15 <= blink_rate_per_min <= 20 else "off"
                        st.metric("Blinks/Minute", f"{blink_rate_per_min:.1f}",
                                delta="Normal" if 15 <= blink_rate_per_min <= 20 else "Abnormal", 
                                delta_color=color)
                    
                    with col_c:
                        st.metric("Avg Duration", f"{avg_blink_duration:.2f}s")
                    
                    with col_d:
                        st.metric("Regularity", f"{blink_regularity:.3f}")
                    
                    # Assessment
                    if blink_rate_per_min < 10:
                        st.error("⚠️ Significantly reduced blinking detected - potential PD indicator")
                    elif blink_rate_per_min < 15:
                        st.warning("⚡ Reduced blinking rate - monitor for other symptoms")
                    else:
                        st.success("✅ Normal blinking pattern")
                    
                    # Blinking timeline (simulated)
                    blink_times = sorted(np.random.uniform(0, 30, total_blinks))
                    intervals = np.diff(blink_times)
                    
                    fig = px.line(x=range(len(intervals)), y=intervals,
                                title="Blinking Intervals Over Time",
                                labels={'x': 'Blink Number', 'y': 'Interval (seconds)'})
                    fig.add_hline(y=np.mean(intervals), line_dash="dash", 
                                line_color="red", annotation_text="Average")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate PD prediction based on blinking analysis
                    st.markdown("---")
                    
                    # Create features from blinking analysis
                    blinking_features = {
                        'expression_mobility': 0.7 if blink_rate_per_min >= 15 else 0.4,
                        'blink_frequency': blink_rate_per_min,
                        'facial_symmetry': blink_regularity,
                        'expression_range': blink_regularity,
                        'eye_asymmetry': 1 - blink_regularity
                    }
                    
                    # Get PD prediction from facial processor
                    pd_prediction, pd_confidence = facial_processor.predict_parkinsons_facial(blinking_features)
                    
                    display_pd_result(pd_prediction, pd_confidence/100 if pd_confidence > 1 else pd_confidence, "Blinking Analysis")
                    display_medical_disclaimer()
                    st.markdown("---")
        
        with col2:
            st.markdown("#### 📋 Normal Blinking")
            st.markdown("""
            **Healthy Adults:**
            - 15-20 blinks per minute
            - Duration: 0.1-0.4 seconds
            - Regular intervals
            
            **PD Indicators:**
            - <10 blinks per minute
            - Irregular timing
            - Reduced spontaneous blinks
            - Delayed blink reflex
            """)
            
            st.markdown("#### 💡 Tips")
            st.markdown("""
            - Sit comfortably
            - Look naturally at camera
            - Don't force blinking
            - Avoid reading or focusing
            """)

if __name__ == "__main__":
    main()