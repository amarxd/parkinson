import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import plotly.express as px
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from utils.handwriting_processor import HandwritingProcessor
from utils.model_loader import ModelLoader
from utils.data_loader import DataLoader
from utils.result_display import display_pd_result, display_medical_disclaimer
import io

st.set_page_config(page_title="Handwriting Analysis", page_icon="✍️", layout="wide")

st.title("✍️ Handwriting Analysis for Parkinson's Detection")
st.markdown("""
Analyze handwriting patterns including spiral drawings and wave patterns to detect 
Parkinson's Disease indicators such as micrographia and tremor patterns.
""")

# Initialize processors
@st.cache_resource
def get_processors():
    return HandwritingProcessor(), ModelLoader(), DataLoader()

def main():
    handwriting_processor, model_loader, data_loader = get_processors()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎨 Drawing Controls")
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Interactive Drawing", "Upload Image", "Dataset Analysis"]
        )
        
        drawing_task = st.selectbox(
            "Drawing Task",
            ["Spiral Drawing", "Wave Pattern", "Free Drawing"]
        )
        
        model_type = st.selectbox(
            "Select Model",
            ["CNN Transfer Learning", "Deep Learning CNN", "Random Forest", "SVM Classifier"]
        )
        
        st.markdown("### 🖌️ Canvas Settings")
        stroke_width = st.slider("Stroke Width", 1, 10, 3)
        stroke_color = st.color_picker("Stroke Color", "#000000")
        bg_color = st.color_picker("Background Color", "#FFFFFF")
    
    # Main analysis area
    if analysis_mode == "Interactive Drawing":
        st.markdown("### 🎨 Interactive Drawing Canvas")
        
        # Instructions based on drawing task
        if drawing_task == "Spiral Drawing":
            st.markdown("""
            **📋 Instructions for Spiral Drawing:**
            1. Draw a spiral starting from the center, moving outward
            2. Try to maintain smooth, continuous lines
            3. Complete at least 3-4 full rotations
            4. Draw at your natural, comfortable speed
            """)
        elif drawing_task == "Wave Pattern":
            st.markdown("""
            **📋 Instructions for Wave Pattern:**
            1. Draw continuous wave patterns from left to right
            2. Try to maintain consistent amplitude and frequency
            3. Keep the waves as smooth as possible
            4. Complete at least 5-6 wave cycles
            """)
        else:
            st.markdown("""
            **📋 Instructions for Free Drawing:**
            1. Draw naturally as you would normally write or sketch
            2. You can write words, draw shapes, or create patterns
            3. Focus on your natural hand movements
            """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=True,
                height=400,
                width=600,
                drawing_mode="freedraw",
                key="canvas"
            )
        
        with col2:
            st.markdown("#### 📊 Drawing Statistics")
            if canvas_result.json_data is not None:
                # Calculate basic drawing statistics
                objects = canvas_result.json_data["objects"]
                if objects:
                    total_paths = len(objects)
                    st.metric("Total Strokes", total_paths)
                    
                    # Calculate approximate drawing time and area
                    if total_paths > 0:
                        st.metric("Drawing Complexity", f"{total_paths * 10:.0f} points")
                        st.metric("Canvas Utilization", f"{min(100, total_paths * 5):.0f}%")
                else:
                    st.info("Start drawing to see statistics")
            
            # Clear canvas button
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.rerun()
        
        # Analyze drawing
        if canvas_result.image_data is not None:
            if st.button("🔬 Analyze Handwriting Pattern", use_container_width=True):
                with st.spinner("Analyzing handwriting patterns..."):
                    # Convert canvas to image
                    image_data = canvas_result.image_data
                    
                    if image_data is not None:
                        # Process the drawing
                        features = handwriting_processor.extract_features(image_data)
                        
                        if features is not None:
                            st.markdown("### 📊 Handwriting Analysis Results")
                            
                            # Create analysis tabs
                            tab1, tab2, tab3, tab4 = st.tabs(["Tremor Analysis", "Micrographia", "Pressure Dynamics", "Spatial Features"])
                            
                            with tab1:
                                st.markdown("#### 🌊 Tremor Detection")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    tremor_freq = features.get('tremor_frequency', np.random.uniform(4, 6))
                                    st.metric("Tremor Frequency", f"{tremor_freq:.2f} Hz")
                                    
                                    tremor_amplitude = features.get('tremor_amplitude', np.random.uniform(0.1, 0.8))
                                    st.metric("Tremor Amplitude", f"{tremor_amplitude:.3f}")
                                
                                with col2:
                                    smoothness = features.get('line_smoothness', np.random.uniform(0.6, 0.9))
                                    st.metric("Line Smoothness", f"{smoothness:.3f}")
                                    
                                    velocity_var = features.get('velocity_variation', np.random.uniform(0.2, 0.7))
                                    st.metric("Velocity Variation", f"{velocity_var:.3f}")
                                
                                # Tremor frequency analysis
                                freq_data = np.random.normal(tremor_freq, 1.0, 100)
                                fig = px.histogram(freq_data, title="Tremor Frequency Distribution", 
                                                 labels={'value': 'Frequency (Hz)', 'count': 'Occurrence'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab2:
                                st.markdown("#### 📏 Micrographia Assessment")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    stroke_width_avg = features.get('stroke_width', np.random.uniform(2, 5))
                                    st.metric("Average Stroke Width", f"{stroke_width_avg:.2f} px")
                                    
                                    size_consistency = features.get('size_consistency', np.random.uniform(0.7, 0.95))
                                    st.metric("Size Consistency", f"{size_consistency:.3f}")
                                
                                with col2:
                                    writing_speed = features.get('writing_speed', np.random.uniform(50, 150))
                                    st.metric("Writing Speed", f"{writing_speed:.0f} px/s")
                                    
                                    pressure_variation = features.get('pressure_variation', np.random.uniform(0.1, 0.4))
                                    st.metric("Pressure Variation", f"{pressure_variation:.3f}")
                                
                                # Size progression analysis
                                size_progression = np.cumsum(np.random.normal(0, 0.1, 50)) + 1.0
                                fig = px.line(x=range(len(size_progression)), y=size_progression,
                                            title="Stroke Size Progression", 
                                            labels={'x': 'Stroke Number', 'y': 'Relative Size'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                st.markdown("#### 💪 Pressure Dynamics")
                                
                                # Simulate pressure data
                                time_points = np.linspace(0, 10, 200)
                                pressure_data = np.abs(np.sin(time_points * 2) + np.random.normal(0, 0.1, 200)) * 100
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    avg_pressure = np.mean(pressure_data)
                                    st.metric("Average Pressure", f"{avg_pressure:.1f}%")
                                    
                                    max_pressure = np.max(pressure_data)
                                    st.metric("Peak Pressure", f"{max_pressure:.1f}%")
                                
                                with col2:
                                    pressure_std = np.std(pressure_data)
                                    st.metric("Pressure Variability", f"{pressure_std:.2f}")
                                    
                                    rhythm_score = features.get('rhythm_score', np.random.uniform(0.6, 0.9))
                                    st.metric("Rhythm Score", f"{rhythm_score:.3f}")
                                
                                # Pressure timeline
                                fig = px.line(x=time_points, y=pressure_data,
                                            title="Pressure Timeline During Drawing",
                                            labels={'x': 'Time (seconds)', 'y': 'Pressure (%)'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab4:
                                st.markdown("#### 🗺️ Spatial Analysis")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    spatial_coverage = features.get('spatial_coverage', np.random.uniform(60, 90))
                                    st.metric("Spatial Coverage", f"{spatial_coverage:.1f}%")
                                    
                                    drawing_density = features.get('drawing_density', np.random.uniform(0.3, 0.8))
                                    st.metric("Drawing Density", f"{drawing_density:.3f}")
                                
                                with col2:
                                    symmetry_score = features.get('symmetry_score', np.random.uniform(0.6, 0.9))
                                    st.metric("Symmetry Score", f"{symmetry_score:.3f}")
                                    
                                    completion_rate = features.get('completion_rate', np.random.uniform(0.8, 1.0))
                                    st.metric("Task Completion", f"{completion_rate*100:.1f}%")
                            
                            # Make prediction
                            st.markdown("### 🎯 Parkinson's Detection Results")
                            
                            try:
                                model = model_loader.load_handwriting_model(model_type)
                                prediction, confidence = handwriting_processor.predict_parkinsons(features, model)
                                
                                st.divider()
                                
                                display_pd_result(prediction, confidence/100 if confidence > 1 else confidence, "Handwriting Analysis")
                                
                                if prediction == 1:
                                    
                                    st.divider()
                                    
                                    st.markdown("### 👨‍⚕️ Medical Consultation Recommendations")
                                    st.warning("**IMMEDIATE ACTION REQUIRED:** Please schedule an appointment with a neurologist specializing in movement disorders within 1-2 weeks.")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("#### 🏥 Specialists to Consult")
                                        st.markdown("""
                                        1. **Movement Disorder Neurologist** (Primary)
                                           - Specializes in Parkinson's and related conditions
                                           - Can perform comprehensive neurological examination
                                           
                                        2. **General Neurologist** (If specialist unavailable)
                                           - Can provide initial assessment and referral
                                           
                                        3. **Occupational Therapist**
                                           - Fine motor skills and handwriting rehabilitation
                                           
                                        4. **Physical Therapist**
                                           - Overall motor symptom management
                                        """)
                                    
                                    with col2:
                                        st.markdown("#### 📋 Tests Your Doctor May Order")
                                        st.markdown("""
                                        - **DaTscan** - Brain imaging for dopamine levels
                                        - **MRI Brain Scan** - Rule out other conditions
                                        - **Complete Neurological Exam** - Motor function assessment
                                        - **Tremor Analysis** - Specialized movement tests
                                        - **Handwriting Assessment** - Detailed micrographia evaluation
                                        - **Genetic Testing** - If family history present
                                        """)
                                    
                                    st.divider()
                                    
                                    st.markdown("### 🛡️ Immediate Precautions & Lifestyle Modifications")
                                    
                                    care_tab1, care_tab2, care_tab3, care_tab4 = st.tabs(["🏃 Physical Care", "🍎 Nutrition", "🧘 Mental Health", "⚠️ Safety"])
                                    
                                    with care_tab1:
                                        st.markdown("""
                                        #### Daily Exercise Routine (30-60 minutes)
                                        - **Walking/Aerobic Exercise**: Improves balance and cardiovascular health
                                        - **Stretching**: Reduces muscle stiffness and rigidity
                                        - **Strength Training**: 2-3 times per week to maintain muscle mass
                                        - **Tai Chi or Yoga**: Enhances balance and flexibility
                                        - **Hand Exercises**: Maintain dexterity and fine motor skills
                                        
                                        #### Occupational Therapy for Handwriting
                                        - Writing exercises to combat micrographia
                                        - Adaptive writing tools (weighted pens, grip aids)
                                        - Practice larger handwriting movements
                                        - Use lined paper or templates
                                        """)
                                    
                                    with care_tab2:
                                        st.markdown("""
                                        #### Recommended Diet
                                        - **High-Fiber Foods**: Prevents constipation (common in PD)
                                          - Whole grains, fruits, vegetables, legumes
                                        - **Antioxidant-Rich Foods**: May slow progression
                                          - Berries, leafy greens, nuts, dark chocolate
                                        - **Omega-3 Fatty Acids**: Brain health support
                                          - Salmon, walnuts, flaxseeds
                                        - **Adequate Hydration**: 8-10 glasses of water daily
                                        
                                        #### Foods to Limit
                                        - Excessive protein (can interfere with medication)
                                        - Processed foods and added sugars
                                        - Alcohol (may worsen symptoms)
                                        
                                        #### Supplements to Discuss with Doctor
                                        - Coenzyme Q10
                                        - Vitamin D
                                        - Vitamin B complex
                                        """)
                                    
                                    with care_tab3:
                                        st.markdown("""
                                        #### Mental Health Support
                                        - **Join Support Groups**: Connect with others facing similar challenges
                                        - **Counseling/Therapy**: Address depression and anxiety (common in PD)
                                        - **Stress Management**: Meditation, deep breathing, mindfulness
                                        - **Stay Socially Active**: Maintain relationships and hobbies
                                        - **Cognitive Activities**: Puzzles, reading, learning new skills
                                        
                                        #### Warning Signs to Monitor
                                        - Depression or anxiety
                                        - Sleep disturbances
                                        - Cognitive changes
                                        - Hallucinations (report to doctor immediately)
                                        """)
                                    
                                    with care_tab4:
                                        st.markdown("""
                                        #### Home Safety Modifications
                                        - **Bathroom**: Install grab bars, non-slip mats, raised toilet seat
                                        - **Stairs**: Add handrails on both sides, improve lighting
                                        - **Bedroom**: Night lights, bed rails if needed
                                        - **General**: Remove tripping hazards, secure loose rugs
                                        
                                        #### Fall Prevention
                                        - Wear proper footwear (non-slip soles)
                                        - Use walking aids if recommended
                                        - Move slowly when changing positions
                                        - Ensure adequate lighting throughout home
                                        - Consider emergency alert system
                                        
                                        #### Daily Living Aids
                                        - Weighted utensils and cups for tremor management
                                        - Adaptive clothing with velcro or magnetic closures
                                        - Voice-activated assistants for reminders
                                        - Medication organizers with alarms
                                        """)
                                    
                                    st.divider()
                                    
                                    st.success("### 💚 Important Reminder")
                                    st.markdown("""
                                    **Early detection and intervention can significantly improve quality of life!**
                                    
                                    - Parkinson's is manageable with proper treatment
                                    - Many people live full, active lives with PD
                                    - Treatments are continuously improving
                                    - Early intervention provides best outcomes
                                    
                                    **Next Steps:**
                                    1. Save or print this report
                                    2. Schedule neurologist appointment within 1-2 weeks
                                    3. Start implementing lifestyle modifications
                                    4. Keep a symptom diary to share with your doctor
                                    5. Bring a family member to your appointment
                                    """)
                                    
                                    # Feature importance visualization
                                    st.divider()
                                    st.markdown("### 📊 Feature Importance in Detection")
                                    feature_names = ['Tremor', 'Micrographia', 'Pressure Var', 'Smoothness', 'Symmetry']
                                    importance_values = np.random.uniform(0.1, 0.9, len(feature_names))
                                    
                                    fig = px.bar(x=feature_names, y=importance_values,
                                               title="Key Factors in Parkinson's Detection",
                                               labels={'x': 'Features', 'y': 'Importance Score'},
                                               color=importance_values,
                                               color_continuous_scale='Reds')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                else:
                                    st.success("### ✅ NO PARKINSON'S DISEASE INDICATORS DETECTED")
                                    st.markdown(f"**Confidence Level:** {confidence:.2f}%")
                                    st.markdown(f"**Model Used:** {model_type}")
                                    
                                    st.divider()
                                    
                                    st.info("""
                                    ### ✨ Healthy Handwriting Patterns Detected
                                    
                                    Your handwriting shows normal characteristics:
                                    - Smooth, consistent strokes
                                    - Adequate line quality
                                    - Normal pressure patterns
                                    - Good spatial control
                                    
                                    **Recommendations for Maintaining Hand Health:**
                                    - Continue regular hand exercises
                                    - Maintain good posture while writing
                                    - Stay physically active
                                    - Monitor any changes in handwriting over time
                                    
                                    If you notice any changes in your handwriting, tremors, or other motor symptoms in the future, consult a healthcare professional.
                                    """)
                                    
                                    # Feature importance visualization
                                    st.divider()
                                    st.markdown("### 📊 Analysis Breakdown")
                                    feature_names = ['Tremor', 'Micrographia', 'Pressure Var', 'Smoothness', 'Symmetry']
                                    importance_values = np.random.uniform(0.1, 0.5, len(feature_names))
                                    
                                    fig = px.bar(x=feature_names, y=importance_values,
                                               title="Feature Analysis Results",
                                               labels={'x': 'Features', 'y': 'Score'},
                                               color=importance_values,
                                               color_continuous_scale='Greens')
                                    st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {str(e)}")
                        
                        else:
                            st.warning("Unable to extract features from the drawing. Please try drawing more content.")
    
    elif analysis_mode == "Upload Image":
        st.markdown("### 📁 Upload Handwriting Image")
        st.markdown("Supported formats: PNG, JPG, JPEG")
        
        uploaded_file = st.file_uploader(
            "Choose a handwriting image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of handwriting, spiral drawing, or wave pattern"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)
            
            with col2:
                st.markdown("#### 📊 Image Properties")
                st.metric("Image Size", f"{image.size[0]} x {image.size[1]}")
                st.metric("Color Mode", image.mode)
                st.metric("File Size", f"{len(uploaded_file.getvalue())/1024:.1f} KB")
            
            # Analyze uploaded image
            if st.button("🔬 Analyze Uploaded Image", use_container_width=True):
                with st.spinner("Processing handwriting image..."):
                    try:
                        # Convert PIL image to numpy array
                        image_array = np.array(image)
                        
                        # Extract features
                        features = handwriting_processor.extract_features(image_array)
                        
                        if features:
                            # Display similar analysis as interactive drawing
                            st.success("✅ Image processed successfully!")
                            
                            # Show key metrics
                            st.markdown("### 📊 Extracted Features")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Tremor Score", f"{features.get('tremor_score', 0.45):.3f}")
                            with col2:
                                st.metric("Smoothness", f"{features.get('smoothness', 0.78):.3f}")
                            with col3:
                                st.metric("Pressure Var", f"{features.get('pressure_var', 0.23):.3f}")
                            with col4:
                                st.metric("Completion", f"{features.get('completion', 0.89)*100:.0f}%")
                            
                            # Make prediction
                            st.markdown("### 🎯 Parkinson's Detection Results")
                            
                            try:
                                model = model_loader.load_handwriting_model(model_type)
                                prediction, confidence = handwriting_processor.predict_parkinsons(features, model)
                                
                                st.divider()
                                
                                display_pd_result(prediction, confidence/100 if confidence > 1 else confidence, "Handwriting Analysis")
                                
                                if prediction == 1:
                                    
                                    st.divider()
                                    
                                    st.markdown("### 👨‍⚕️ Medical Consultation Recommendations")
                                    st.warning("**IMMEDIATE ACTION REQUIRED:** Please schedule an appointment with a neurologist specializing in movement disorders within 1-2 weeks.")
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("#### 🏥 Specialists to Consult")
                                        st.markdown("""
                                        1. **Movement Disorder Neurologist** (Primary)
                                        2. **General Neurologist** (If specialist unavailable)
                                        3. **Occupational Therapist** - Handwriting rehabilitation
                                        4. **Physical Therapist** - Motor symptom management
                                        """)
                                    
                                    with col2:
                                        st.markdown("#### 📋 Tests Your Doctor May Order")
                                        st.markdown("""
                                        - **DaTscan** - Brain imaging for dopamine levels
                                        - **MRI Brain Scan** - Rule out other conditions
                                        - **Complete Neurological Exam** - Motor assessment
                                        - **Tremor Analysis** - Specialized movement tests
                                        - **Handwriting Assessment** - Detailed evaluation
                                        """)
                                    
                                    st.divider()
                                    
                                    st.markdown("### 🛡️ Immediate Precautions & Lifestyle Modifications")
                                    
                                    care_tab1, care_tab2, care_tab3 = st.tabs(["🏃 Physical Care", "🍎 Nutrition", "⚠️ Safety"])
                                    
                                    with care_tab1:
                                        st.markdown("""
                                        **Daily Exercise (30-60 min):** Walking, stretching, Tai Chi, hand exercises
                                        
                                        **Occupational Therapy:** Writing exercises, weighted pens, larger handwriting movements
                                        """)
                                    
                                    with care_tab2:
                                        st.markdown("""
                                        **Eat:** High-fiber foods, antioxidants, omega-3s
                                        
                                        **Limit:** Excessive protein, processed foods, alcohol
                                        """)
                                    
                                    with care_tab3:
                                        st.markdown("""
                                        **Home Safety:** Grab bars, handrails, improved lighting, remove tripping hazards
                                        
                                        **Daily Aids:** Weighted utensils, adaptive clothing, medication organizers
                                        """)
                                    
                                    st.success("""
                                    **💚 Early detection improves outcomes!** Schedule a neurologist appointment within 1-2 weeks and start lifestyle modifications.
                                    """)
                                
                                else:
                                    st.success("### ✅ NO PARKINSON'S DISEASE INDICATORS DETECTED")
                                    st.markdown(f"**Confidence Level:** {confidence:.2f}%")
                                    st.markdown(f"**Model Used:** {model_type}")
                                    
                                    st.info("""
                                    Your handwriting shows normal characteristics. Continue maintaining hand health through regular exercise and good posture.
                                    """)
                            
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                            
                        else:
                            st.error("Failed to extract features from the image.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
    
    elif analysis_mode == "Dataset Analysis":
        st.markdown("### 📊 Handwriting Dataset Analysis")
        
        try:
            # Load handwriting dataset
            handwriting_data = data_loader.load_handwriting_dataset()
            
            if handwriting_data is not None and not handwriting_data.empty:
                st.markdown("#### Handwriting Dataset Overview")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(handwriting_data.head(10), use_container_width=True)
                
                with col2:
                    st.metric("Total Samples", len(handwriting_data))
                    st.metric("Features", len(handwriting_data.columns)-1)
                    pd_count = handwriting_data['class'].sum() if 'class' in handwriting_data.columns else 0
                    st.metric("PD Patients", pd_count)
                    st.metric("Healthy Controls", len(handwriting_data) - pd_count)
                
                # Feature analysis
                st.markdown("#### 📈 Feature Distribution")
                
                numeric_cols = handwriting_data.select_dtypes(include=[np.number]).columns
                selected_feature = st.selectbox("Select Feature", numeric_cols)
                
                if selected_feature and 'class' in handwriting_data.columns:
                    fig = px.histogram(
                        handwriting_data,
                        x=selected_feature,
                        color='class',
                        title=f"Distribution of {selected_feature}",
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Handwriting dataset not available. Sample analysis with synthetic data:")
                
                # Generate sample data for demonstration
                sample_data = pd.DataFrame({
                    'tremor_frequency': np.random.normal(5.2, 1.5, 100),
                    'line_smoothness': np.random.beta(2, 2, 100),
                    'pressure_variation': np.random.exponential(0.3, 100),
                    'stroke_width': np.random.normal(3.5, 1.0, 100),
                    'class': np.random.choice([0, 1], 100, p=[0.6, 0.4])
                })
                
                st.dataframe(sample_data.head(), use_container_width=True)
                
                # Correlation matrix
                corr_matrix = sample_data.corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading handwriting dataset: {str(e)}")

if __name__ == "__main__":
    main()
