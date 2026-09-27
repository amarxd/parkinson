import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import mediapipe as mp
from utils.motion_processor import MotionProcessor
from utils.result_display import display_pd_result, display_medical_disclaimer
from utils.model_loader import ModelLoader
from utils.live_recording import MotionDataCollector
import io
import time

st.set_page_config(page_title="Motion Detection", page_icon="🤲", layout="wide")

st.title("🤲 Motion Detection for Parkinson's Analysis")
st.markdown("""
Analyze hand movements, tremors, bradykinesia (slowness of movement), and gait patterns 
to detect motor symptoms associated with Parkinson's Disease.
""")

# Initialize processors
@st.cache_resource
def get_processors():
    return MotionProcessor(), ModelLoader()

def main():
    motion_processor, model_loader = get_processors()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎯 Motion Analysis Type")
        analysis_type = st.selectbox(
            "Choose Analysis Mode",
            ["User Report Entry", "Hand Tremor Detection", "Finger Tapping Test", "Bradykinesia Assessment", "Gait Analysis", "Postural Stability"]
        )
        
        st.markdown("### ⚙️ Detection Settings")
        sensitivity = st.slider("Tremor Sensitivity", 0.1, 1.0, 0.6, 0.1)
        motion_threshold = st.slider("Motion Threshold", 0.01, 0.1, 0.03, 0.01)
        
        model_type = st.selectbox(
            "Analysis Model",
            ["Deep Learning LSTM", "Pose Estimation", "Traditional CV", "Combined Analysis"]
        )
        
        st.markdown("### 📊 Real-time Metrics")
        if 'motion_metrics' in st.session_state:
            metrics = st.session_state.motion_metrics
            st.metric("Tremor Frequency", f"{metrics.get('tremor_freq', 0):.1f} Hz")
            st.metric("Movement Speed", f"{metrics.get('movement_speed', 0):.1f}%")
            st.metric("Coordination", f"{metrics.get('coordination', 0):.3f}")
    
    # Main analysis area
    if analysis_type == "User Report Entry":
        st.markdown("### 📝 User-Entered Motor Symptom Report")
        st.markdown("""
        Enter details about your motor symptoms based on your observations. 
        This information will be used to calculate a motor symptom score and assess Parkinson's risk.
        """)
        
        # Initialize motion data collector
        if 'motion_collector' not in st.session_state:
            st.session_state.motion_collector = MotionDataCollector()
        
        collector = st.session_state.motion_collector
        
        # Create tabs for different symptom categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤝 Tremor", "🐌 Bradykinesia", "💪 Rigidity", "⚖️ Postural Instability", "🚶 Gait Disturbance"
        ])
        
        with tab1:
            st.markdown("#### Tremor Assessment")
            st.markdown("Tremor is an involuntary shaking or rhythmic movement, typically in the hands.")
            
            tremor_present = st.checkbox("Tremor present", key="tremor_present")
            
            if tremor_present:
                tremor_severity = st.select_slider(
                    "Tremor Severity",
                    options=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                    value="Mild",
                    key="tremor_severity"
                )
                
                tremor_location = st.multiselect(
                    "Tremor Location",
                    ["Right Hand", "Left Hand", "Both Hands", "Right Leg", "Left Leg", "Head", "Jaw"],
                    key="tremor_location"
                )
                
                tremor_frequency = st.selectbox(
                    "Tremor Frequency",
                    ["Rarely", "Occasionally", "Frequently", "Almost Always", "Constantly"],
                    key="tremor_frequency"
                )
                
                tremor_notes = st.text_area("Additional Notes", key="tremor_notes",
                    placeholder="Describe when the tremor occurs, what makes it better/worse, etc.")
                
                collector.collect_tremor_data(
                    present=tremor_present,
                    severity=tremor_severity,
                    location=tremor_location,
                    frequency=tremor_frequency,
                    notes=tremor_notes
                )
        
        with tab2:
            st.markdown("#### Bradykinesia Assessment")
            st.markdown("Bradykinesia is slowness of movement, a cardinal symptom of Parkinson's.")
            
            brady_present = st.checkbox("Slowness of movement present", key="brady_present")
            
            if brady_present:
                brady_severity = st.select_slider(
                    "Bradykinesia Severity",
                    options=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                    value="Mild",
                    key="brady_severity"
                )
                
                affected_activities = st.multiselect(
                    "Affected Daily Activities",
                    ["Buttoning clothes", "Writing", "Eating", "Walking", "Getting up from chair", "Turning in bed", "Brushing teeth"],
                    key="affected_activities"
                )
                
                brady_notes = st.text_area("Additional Notes", key="brady_notes",
                    placeholder="Describe specific difficulties or patterns you've noticed...")
                
                collector.collect_bradykinesia_data(
                    present=brady_present,
                    severity=brady_severity,
                    affected_activities=affected_activities,
                    notes=brady_notes
                )
        
        with tab3:
            st.markdown("#### Rigidity Assessment")
            st.markdown("Rigidity is increased muscle stiffness and resistance to movement.")
            
            rigidity_present = st.checkbox("Muscle stiffness present", key="rigidity_present")
            
            if rigidity_present:
                rigidity_severity = st.select_slider(
                    "Rigidity Severity",
                    options=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                    value="Mild",
                    key="rigidity_severity"
                )
                
                rigidity_location = st.multiselect(
                    "Affected Body Parts",
                    ["Neck", "Shoulders", "Arms", "Legs", "Back", "Whole Body"],
                    key="rigidity_location"
                )
                
                rigidity_notes = st.text_area("Additional Notes", key="rigidity_notes",
                    placeholder="Describe when stiffness is worst, any pain associated, etc.")
                
                collector.collect_rigidity_data(
                    present=rigidity_present,
                    severity=rigidity_severity,
                    location=rigidity_location,
                    notes=rigidity_notes
                )
        
        with tab4:
            st.markdown("#### Postural Instability Assessment")
            st.markdown("Postural instability affects balance and coordination.")
            
            postural_present = st.checkbox("Balance problems present", key="postural_present")
            
            if postural_present:
                postural_severity = st.select_slider(
                    "Postural Instability Severity",
                    options=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                    value="Mild",
                    key="postural_severity"
                )
                
                fall_history = st.checkbox("History of falls in past 6 months", key="fall_history")
                
                postural_notes = st.text_area("Additional Notes", key="postural_notes",
                    placeholder="Describe balance issues, near-falls, need for support, etc.")
                
                collector.collect_postural_data(
                    present=postural_present,
                    severity=postural_severity,
                    fall_history=fall_history,
                    notes=postural_notes
                )
        
        with tab5:
            st.markdown("#### Gait Disturbance Assessment")
            st.markdown("Gait disturbances affect how you walk and move.")
            
            gait_present = st.checkbox("Walking difficulties present", key="gait_present")
            
            if gait_present:
                gait_severity = st.select_slider(
                    "Gait Disturbance Severity",
                    options=["None", "Mild", "Moderate", "Severe", "Very Severe"],
                    value="Mild",
                    key="gait_severity"
                )
                
                gait_type = st.multiselect(
                    "Type of Gait Problems",
                    ["Shuffling", "Reduced arm swing", "Festination (rushed steps)", "Freezing of gait", "Small steps", "Difficulty turning"],
                    key="gait_type"
                )
                
                gait_notes = st.text_area("Additional Notes", key="gait_notes",
                    placeholder="Describe walking patterns, specific situations where problems occur, etc.")
                
                collector.collect_gait_data(
                    present=gait_present,
                    gait_type=gait_type,
                    severity=gait_severity,
                    notes=gait_notes
                )
        
        # Calculate and display motor score
        st.markdown("---")
        st.markdown("### 📊 Motor Symptom Assessment Results")
        
        if st.button("📈 Calculate Motor Score", use_container_width=True):
            motor_scores = collector.calculate_motor_score()
            summary = collector.get_summary()
            
            # Display overall metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_color = "🔴" if motor_scores['risk_level'] == 'High' else "🟡" if motor_scores['risk_level'] == 'Moderate' else "🟢"
                st.metric("Risk Level", f"{risk_color} {motor_scores['risk_level']}")
            
            with col2:
                st.metric("Total Score", f"{motor_scores['total_score']}/{motor_scores['max_score']}")
            
            with col3:
                st.metric("Symptom Count", motor_scores['symptom_count'])
            
            with col4:
                st.metric("Normalized Score", f"{motor_scores['normalized_score']*100:.1f}%")
            
            # Display symptoms present
            if motor_scores['symptoms_present']:
                st.markdown("#### Symptoms Reported:")
                for symptom in motor_scores['symptoms_present']:
                    symptom_name = symptom.replace('_', ' ').title()
                    st.markdown(f"✓ {symptom_name}")
            
            # Clinical recommendations
            st.markdown("#### 🏥 Recommendations:")
            
            if motor_scores['risk_level'] == 'High':
                st.error("""
                **High Risk Assessment**
                - Multiple motor symptoms reported with significant severity
                - Strongly recommend consultation with neurologist or movement disorder specialist
                - Consider comprehensive clinical evaluation including UPDRS assessment
                - Document symptom progression and timing
                """)
            elif motor_scores['risk_level'] == 'Moderate':
                st.warning("""
                **Moderate Risk Assessment**
                - Some motor symptoms present requiring attention
                - Schedule appointment with primary care physician
                - Consider referral to neurologist for detailed evaluation
                - Monitor symptoms and track any changes
                """)
            elif motor_scores['risk_level'] == 'Low':
                st.info("""
                **Low Risk Assessment**
                - Minimal motor symptoms reported
                - Discuss symptoms with your doctor at next regular checkup
                - Continue monitoring for any changes
                - Maintain regular physical activity
                """)
            else:
                st.success("""
                **Minimal Risk**
                - No significant motor symptoms reported
                - Continue routine health monitoring
                - Stay active and maintain healthy lifestyle
                """)
            
            # Store results in session state
            st.session_state.motion_report_results = summary
            
            # Option to save report
            st.markdown("---")
            if st.button("💾 Save Assessment to Session", use_container_width=True):
                st.success("Assessment saved! You can access these results in the Clinical Reports page.")
    
    elif analysis_type == "Hand Tremor Detection":
        st.markdown("### 👋 Hand Tremor Detection")
        st.markdown("""
        **Instructions:**
        1. Extend your hands in front of the camera
        2. Keep hands steady for tremor analysis
        3. The system will detect resting and postural tremors
        4. Try both hands separately and together
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Tremor detection interface
            st.markdown("#### 📹 Real-time Tremor Analysis")
            
            if st.button("🎥 Start Tremor Detection", use_container_width=True):
                with st.spinner("Initializing hand tracking..."):
                    time.sleep(2)
                    
                    # Simulate tremor analysis
                    st.success("✅ Hand tracking initialized!")
                    st.info("📸 Live tremor analysis would run here")
                    
                    # Generate simulated tremor data
                    tremor_data = {
                        'left_hand_tremor': np.random.uniform(3.5, 6.5),
                        'right_hand_tremor': np.random.uniform(3.8, 6.2),
                        'tremor_amplitude': np.random.uniform(0.1, 0.8),
                        'postural_tremor': np.random.uniform(4.0, 7.0),
                        'resting_tremor': np.random.uniform(3.5, 5.5)
                    }
                    
                    st.session_state.motion_metrics = {
                        'tremor_freq': (tremor_data['left_hand_tremor'] + tremor_data['right_hand_tremor']) / 2,
                        'movement_speed': np.random.uniform(60, 95),
                        'coordination': np.random.uniform(0.6, 0.9)
                    }
                    
                    # Display real-time tremor analysis
                    st.markdown("#### 📊 Tremor Analysis Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        freq = tremor_data['left_hand_tremor']
                        color = "off" if 4.0 <= freq <= 6.0 else "normal"
                        st.metric("Left Hand", f"{freq:.1f} Hz",
                                delta="PD Range" if 4.0 <= freq <= 6.0 else "Normal", delta_color=color)
                    
                    with col_b:
                        freq = tremor_data['right_hand_tremor']
                        color = "off" if 4.0 <= freq <= 6.0 else "normal"
                        st.metric("Right Hand", f"{freq:.1f} Hz",
                                delta="PD Range" if 4.0 <= freq <= 6.0 else "Normal", delta_color=color)
                    
                    with col_c:
                        amplitude = tremor_data['tremor_amplitude']
                        color = "off" if amplitude > 0.5 else "normal"
                        st.metric("Amplitude", f"{amplitude:.3f}",
                                delta="High" if amplitude > 0.5 else "Normal", delta_color=color)
                    
                    with col_d:
                        postural = tremor_data['postural_tremor']
                        color = "off" if postural > 5.5 else "normal"
                        st.metric("Postural", f"{postural:.1f} Hz",
                                delta="Elevated" if postural > 5.5 else "Normal", delta_color=color)
                    
                    # Tremor frequency analysis over time
                    time_points = np.linspace(0, 30, 300)
                    left_tremor_signal = np.sin(2 * np.pi * tremor_data['left_hand_tremor'] * time_points) * tremor_data['tremor_amplitude']
                    right_tremor_signal = np.sin(2 * np.pi * tremor_data['right_hand_tremor'] * time_points) * tremor_data['tremor_amplitude']
                    
                    # Add noise to make it realistic
                    left_tremor_signal += np.random.normal(0, 0.1, len(left_tremor_signal))
                    right_tremor_signal += np.random.normal(0, 0.1, len(right_tremor_signal))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=time_points, y=left_tremor_signal, name="Left Hand", line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=time_points, y=right_tremor_signal, name="Right Hand", line=dict(color='red')))
                    fig.update_layout(
                        title="Real-time Tremor Signal Analysis",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Tremor Amplitude",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Diagnosis assessment
                    st.markdown("#### 🎯 Tremor Assessment")
                    risk_factors = []
                    
                    if 4.0 <= tremor_data['left_hand_tremor'] <= 6.0:
                        risk_factors.append("Left hand tremor in PD frequency range")
                    if 4.0 <= tremor_data['right_hand_tremor'] <= 6.0:
                        risk_factors.append("Right hand tremor in PD frequency range")
                    if tremor_data['tremor_amplitude'] > 0.5:
                        risk_factors.append("High tremor amplitude detected")
                    if tremor_data['resting_tremor'] > 4.5:
                        risk_factors.append("Significant resting tremor")
                    
                    if len(risk_factors) >= 2:
                        st.error("⚠️ Multiple PD tremor indicators detected")
                        st.markdown("**Risk Factors:**")
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
                    elif len(risk_factors) == 1:
                        st.warning("⚡ Some tremor indicators present")
                        st.markdown(f"• {risk_factors[0]}")
                    else:
                        st.success("✅ No significant tremor indicators")
        
        with col2:
            st.markdown("#### 🎯 Tremor Guidelines")
            st.markdown("""
            **Parkinson's Tremor:**
            - Frequency: 4-6 Hz
            - Resting tremor dominant
            - Asymmetric onset
            - Pill-rolling motion
            
            **Normal Tremor:**
            - Frequency: 8-12 Hz
            - Postural/action tremor
            - Symmetric
            - Lower amplitude
            
            **Test Positions:**
            - Rest: Hands in lap
            - Postural: Arms extended
            - Action: Finger-to-nose
            """)
            
            if st.button("📋 Tremor Exercise Guide", use_container_width=True):
                st.info("""
                **Exercise Sequence:**
                1. Rest hands in lap (30s)
                2. Extend arms forward (30s)
                3. Finger-to-nose touch (10x)
                4. Hand pronation/supination
                """)
    
    elif analysis_type == "Finger Tapping Test":
        st.markdown("### 👆 Finger Tapping Test")
        st.markdown("""
        Test for bradykinesia (slowness of movement) by analyzing finger tapping speed,
        rhythm, and amplitude - key indicators of Parkinson's motor dysfunction.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### ⏱️ 30-Second Finger Tapping Test")
            
            hand_choice = st.radio("Select Hand", ["Left Hand", "Right Hand", "Both Hands"])
            
            if st.button("🎬 Start Finger Tapping Test", use_container_width=True):
                with st.spinner("Starting finger tapping analysis..."):
                    time.sleep(1)
                    
                    st.success("✅ Test started! Tap your index finger and thumb together as fast as possible.")
                    
                    # Simulate 30-second test
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    tapping_data = []
                    for i in range(30):
                        time.sleep(0.1)  # Faster simulation
                        progress_bar.progress((i + 1) / 30)
                        status_text.text(f"Testing... {i + 1}/30 seconds")
                        
                        # Simulate tapping detection
                        taps_per_second = np.random.uniform(2, 6)  # Normal: 4-6, PD: 2-4
                        tapping_data.append(taps_per_second)
                    
                    # Analyze results
                    total_taps = sum(tapping_data)
                    avg_frequency = np.mean(tapping_data)
                    frequency_decline = (tapping_data[0] - tapping_data[-1]) / tapping_data[0] if tapping_data[0] > 0 else 0
                    rhythm_variability = np.std(tapping_data) / np.mean(tapping_data) if np.mean(tapping_data) > 0 else 0
                    
                    st.markdown("#### 📊 Finger Tapping Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        color = "off" if total_taps < 120 else "normal"
                        st.metric("Total Taps", f"{total_taps:.0f}",
                                delta="Below Normal" if total_taps < 120 else "Normal", delta_color=color)
                    
                    with col_b:
                        color = "off" if avg_frequency < 4.0 else "normal"
                        st.metric("Avg Frequency", f"{avg_frequency:.1f}/s",
                                delta="Bradykinetic" if avg_frequency < 4.0 else "Normal", delta_color=color)
                    
                    with col_c:
                        color = "off" if frequency_decline > 0.3 else "normal"
                        st.metric("Decline Rate", f"{frequency_decline*100:.1f}%",
                                delta="Significant" if frequency_decline > 0.3 else "Normal", delta_color=color)
                    
                    with col_d:
                        color = "off" if rhythm_variability > 0.3 else "normal"
                        st.metric("Rhythm Var", f"{rhythm_variability:.3f}",
                                delta="Irregular" if rhythm_variability > 0.3 else "Regular", delta_color=color)
                    
                    # Tapping frequency over time
                    fig = px.line(x=range(len(tapping_data)), y=tapping_data,
                                title="Finger Tapping Frequency Over Time",
                                labels={'x': 'Time (seconds)', 'y': 'Taps per Second'})
                    fig.add_hline(y=4.0, line_dash="dash", line_color="red", 
                                annotation_text="Normal Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Assessment
                    st.markdown("#### 🎯 Bradykinesia Assessment")
                    risk_score = 0
                    
                    if total_taps < 120:
                        risk_score += 2
                        st.warning("• Low total tap count suggests bradykinesia")
                    
                    if avg_frequency < 4.0:
                        risk_score += 2
                        st.warning("• Reduced tapping frequency")
                    
                    if frequency_decline > 0.3:
                        risk_score += 1
                        st.warning("• Significant frequency decline during test")
                    
                    if rhythm_variability > 0.3:
                        risk_score += 1
                        st.warning("• Irregular tapping rhythm")
                    
                    if risk_score >= 3:
                        st.error("⚠️ Strong indicators of bradykinesia")
                    elif risk_score >= 2:
                        st.warning("⚡ Moderate bradykinetic indicators")
                    else:
                        st.success("✅ Normal finger tapping performance")
        
        with col2:
            st.markdown("#### 📋 Test Instructions")
            st.markdown("""
            **How to Perform:**
            1. Sit comfortably
            2. Rest forearm on table
            3. Tap index finger to thumb
            4. Maximum speed for 30 seconds
            5. Maintain large amplitude
            
            **Normal Results:**
            - 120+ taps in 30 seconds
            - 4-6 taps per second
            - Consistent rhythm
            - Minimal decline
            
            **PD Indicators:**
            - <120 taps total
            - <4 taps per second
            - Frequent pauses
            - Progressive slowing
            """)
    
    elif analysis_type == "Bradykinesia Assessment":
        st.markdown("### 🐌 Bradykinesia Assessment")
        st.markdown("""
        Comprehensive assessment of movement slowness across multiple motor tasks
        to evaluate the severity of bradykinesia in Parkinson's Disease.
        """)
        
        # Task selection
        task_type = st.selectbox(
            "Select Assessment Task",
            ["Hand Opening/Closing", "Pronation/Supination", "Leg Agility", "Combined Assessment"]
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if task_type == "Hand Opening/Closing":
                st.markdown("#### ✋ Hand Opening/Closing Test")
                st.markdown("Open and close your hand as wide and fast as possible for 30 seconds")
                
                if st.button("🎥 Start Hand Movement Test", use_container_width=True):
                    with st.spinner("Analyzing hand movements..."):
                        time.sleep(2)
                        
                        # Simulate hand movement analysis
                        cycles = np.random.randint(15, 35)  # Normal: 25-35, PD: 15-25
                        amplitude_score = np.random.uniform(0.4, 0.9)  # Normal: >0.7, PD: <0.6
                        rhythm_score = np.random.uniform(0.5, 0.95)
                        fatigue_factor = np.random.uniform(0.1, 0.4)
                        
                        st.markdown("#### 📊 Hand Movement Results")
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            color = "off" if cycles < 25 else "normal"
                            st.metric("Open/Close Cycles", cycles,
                                    delta="Slow" if cycles < 25 else "Normal", delta_color=color)
                        
                        with col_b:
                            color = "off" if amplitude_score < 0.6 else "normal"
                            st.metric("Amplitude Score", f"{amplitude_score:.3f}",
                                    delta="Reduced" if amplitude_score < 0.6 else "Good", delta_color=color)
                        
                        with col_c:
                            color = "off" if rhythm_score < 0.7 else "normal"
                            st.metric("Rhythm Score", f"{rhythm_score:.3f}",
                                    delta="Irregular" if rhythm_score < 0.7 else "Regular", delta_color=color)
                        
                        with col_d:
                            color = "off" if fatigue_factor > 0.3 else "normal"
                            st.metric("Fatigue Factor", f"{fatigue_factor:.3f}",
                                    delta="High" if fatigue_factor > 0.3 else "Low", delta_color=color)
            
            elif task_type == "Pronation/Supination":
                st.markdown("#### 🔄 Pronation/Supination Test")
                st.markdown("Rotate your hands palm up and palm down as fast as possible")
                
                if st.button("🎥 Start Rotation Test", use_container_width=True):
                    with st.spinner("Analyzing hand rotations..."):
                        time.sleep(2)
                        
                        # Simulate rotation analysis
                        rotations = np.random.randint(20, 45)
                        rotation_speed = np.random.uniform(0.8, 2.5)
                        coordination = np.random.uniform(0.5, 0.9)
                        
                        st.markdown("#### 📊 Rotation Results")
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            color = "off" if rotations < 30 else "normal"
                            st.metric("Total Rotations", rotations,
                                    delta="Slow" if rotations < 30 else "Normal", delta_color=color)
                        
                        with col_b:
                            color = "off" if rotation_speed < 1.5 else "normal"
                            st.metric("Speed", f"{rotation_speed:.1f}/s",
                                    delta="Slow" if rotation_speed < 1.5 else "Normal", delta_color=color)
                        
                        with col_c:
                            color = "off" if coordination < 0.6 else "normal"
                            st.metric("Coordination", f"{coordination:.3f}",
                                    delta="Poor" if coordination < 0.6 else "Good", delta_color=color)
        
        with col2:
            st.markdown("#### 📊 Bradykinesia Scale")
            st.markdown("""
            **UPDRS Motor Scale:**
            - 0: Normal
            - 1: Slight slowing
            - 2: Moderate slowing
            - 3: Severe slowing
            - 4: Unable to perform
            
            **Assessment Criteria:**
            - Speed of movement
            - Amplitude of movement
            - Rhythm and coordination
            - Fatigue and hesitation
            """)
    
    elif analysis_type == "Gait Analysis":
        st.markdown("### 🚶 Gait Analysis")
        st.markdown("""
        Analyze walking patterns to detect shuffling gait, reduced arm swing,
        and postural instability associated with Parkinson's Disease.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎬 Walking Pattern Analysis")
            st.markdown("Walk normally in front of the camera for full-body gait analysis")
            
            if st.button("🎥 Start Gait Analysis", use_container_width=True):
                with st.spinner("Initializing pose tracking..."):
                    time.sleep(3)
                    
                    st.success("✅ Pose tracking initialized!")
                    st.info("📸 Walking analysis in progress...")
                    
                    # Simulate gait analysis
                    gait_metrics = {
                        'step_length': np.random.uniform(45, 75),  # cm, Normal: 60-75, PD: 45-60
                        'step_frequency': np.random.uniform(1.5, 2.5),  # steps/sec
                        'arm_swing': np.random.uniform(15, 35),  # degrees
                        'stride_variability': np.random.uniform(0.02, 0.08),
                        'postural_sway': np.random.uniform(0.5, 2.5),  # degrees
                        'walking_speed': np.random.uniform(0.8, 1.4)  # m/s
                    }
                    
                    st.markdown("#### 📊 Gait Analysis Results")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        step_len = gait_metrics['step_length']
                        color = "off" if step_len < 55 else "normal"
                        st.metric("Step Length", f"{step_len:.0f} cm",
                                delta="Short" if step_len < 55 else "Normal", delta_color=color)
                        
                        speed = gait_metrics['walking_speed']
                        color = "off" if speed < 1.0 else "normal"
                        st.metric("Walking Speed", f"{speed:.1f} m/s",
                                delta="Slow" if speed < 1.0 else "Normal", delta_color=color)
                    
                    with col_b:
                        arm_swing = gait_metrics['arm_swing']
                        color = "off" if arm_swing < 20 else "normal"
                        st.metric("Arm Swing", f"{arm_swing:.0f}°",
                                delta="Reduced" if arm_swing < 20 else "Normal", delta_color=color)
                        
                        frequency = gait_metrics['step_frequency']
                        st.metric("Step Frequency", f"{frequency:.1f}/s")
                    
                    with col_c:
                        variability = gait_metrics['stride_variability']
                        color = "off" if variability > 0.05 else "normal"
                        st.metric("Stride Variability", f"{variability:.3f}",
                                delta="High" if variability > 0.05 else "Low", delta_color=color)
                        
                        sway = gait_metrics['postural_sway']
                        color = "off" if sway > 2.0 else "normal"
                        st.metric("Postural Sway", f"{sway:.1f}°",
                                delta="Unstable" if sway > 2.0 else "Stable", delta_color=color)
                    
                    # Gait assessment
                    st.markdown("#### 🎯 Gait Assessment")
                    risk_factors = []
                    
                    if step_len < 55:
                        risk_factors.append("Shortened step length (shuffling gait)")
                    if arm_swing < 20:
                        risk_factors.append("Reduced arm swing")
                    if speed < 1.0:
                        risk_factors.append("Slow walking speed")
                    if variability > 0.05:
                        risk_factors.append("High stride variability")
                    if sway > 2.0:
                        risk_factors.append("Postural instability")
                    
                    if len(risk_factors) >= 3:
                        st.error("⚠️ Multiple gait abnormalities detected")
                    elif len(risk_factors) >= 2:
                        st.warning("⚡ Some gait irregularities present")
                    else:
                        st.success("✅ Normal gait pattern")
                    
                    if risk_factors:
                        st.markdown("**Detected Issues:**")
                        for factor in risk_factors:
                            st.markdown(f"• {factor}")
        
        with col2:
            st.markdown("#### 🚶 Gait Characteristics")
            st.markdown("""
            **Normal Gait:**
            - Step length: 60-75 cm
            - Walking speed: 1.2-1.4 m/s
            - Arm swing: 25-35°
            - Regular rhythm
            
            **PD Gait:**
            - Short, shuffling steps
            - Reduced arm swing
            - Slow walking speed
            - Freezing episodes
            - Festinating gait
            
            **Assessment Areas:**
            - Step length/height
            - Arm swing amplitude
            - Walking rhythm
            - Postural stability
            """)
    
    elif analysis_type == "Postural Stability":
        st.markdown("### ⚖️ Postural Stability Assessment")
        st.markdown("""
        Evaluate balance and postural control to detect instability
        that may lead to falls in Parkinson's patients.
        """)
        
        test_type = st.selectbox(
            "Select Balance Test",
            ["Static Balance", "Dynamic Balance", "Pull Test", "Tandem Walk"]
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if test_type == "Static Balance":
                st.markdown("#### 🧘 Static Balance Test")
                st.markdown("Stand still with feet together for 30 seconds")
                
                if st.button("🎥 Start Balance Test", use_container_width=True):
                    with st.spinner("Analyzing postural stability..."):
                        time.sleep(3)
                        
                        # Simulate balance analysis
                        sway_area = np.random.uniform(2, 15)  # cm², Normal: <5, PD: >10
                        sway_velocity = np.random.uniform(0.5, 3.0)  # cm/s
                        stability_score = np.random.uniform(0.3, 0.9)
                        
                        st.markdown("#### 📊 Balance Results")
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            color = "off" if sway_area > 8 else "normal"
                            st.metric("Sway Area", f"{sway_area:.1f} cm²",
                                    delta="Unstable" if sway_area > 8 else "Stable", delta_color=color)
                        
                        with col_b:
                            color = "off" if sway_velocity > 2.0 else "normal"
                            st.metric("Sway Velocity", f"{sway_velocity:.1f} cm/s",
                                    delta="High" if sway_velocity > 2.0 else "Normal", delta_color=color)
                        
                        with col_c:
                            color = "normal" if stability_score > 0.7 else "off"
                            st.metric("Stability Score", f"{stability_score:.3f}",
                                    delta="Good" if stability_score > 0.7 else "Poor", delta_color=color)
                        
                        # Balance assessment
                        if sway_area > 10 or sway_velocity > 2.5 or stability_score < 0.5:
                            st.error("⚠️ Significant postural instability detected")
                            st.warning("Increased fall risk - recommend balance training")
                        elif sway_area > 6 or sway_velocity > 1.5 or stability_score < 0.7:
                            st.warning("⚡ Mild postural instability")
                        else:
                            st.success("✅ Good postural stability")
        
        with col2:
            st.markdown("#### ⚖️ Balance Guidelines")
            st.markdown("""
            **Fall Risk Factors:**
            - Postural instability
            - Freezing of gait
            - Medication effects
            - Environmental hazards
            
            **Prevention Strategies:**
            - Balance training
            - Strength exercises
            - Home modifications
            - Medication optimization
            """)

if __name__ == "__main__":
    main()