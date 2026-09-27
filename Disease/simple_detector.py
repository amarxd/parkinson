import streamlit as st
import numpy as np
from PIL import Image
import io
from utils.voice_processor import VoiceProcessor
from utils.handwriting_processor import HandwritingProcessor
from utils.model_loader import ModelLoader

st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_models():
    model_loader = ModelLoader()
    voice_proc = VoiceProcessor()
    handwriting_proc = HandwritingProcessor()
    return model_loader, voice_proc, handwriting_proc

st.title("🧠 Parkinson's Disease Detector")
st.markdown("### Upload a voice recording or handwriting image to check for Parkinson's Disease")

st.info("""
### 🎯 Quick Start: Test with Sample Files
**To see a POSITIVE Parkinson's Disease detection (with full medical recommendations):**
- **Voice Test**: Upload `parkinsons_voice_sample.wav` 
- **Handwriting Test**: Upload `parkinsons_spiral.png` or `parkinsons_wave.png`

**For negative (healthy) results:**
- **Voice Test**: Upload `healthy_voice_sample.wav`
- **Handwriting Test**: Upload `healthy_spiral.png` or `healthy_wave.png`

📂 All sample files are located in the project root folder and available for download.
""")

model_loader, voice_proc, handwriting_proc = load_models()

tab1, tab2 = st.tabs(["🎙️ Voice Test", "✍️ Handwriting Test"])

with tab1:
    st.markdown("#### Upload a voice recording")
    st.info("📋 Upload a WAV, MP3, or FLAC audio file")
    
    audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'flac'], key='audio')
    
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
            with st.spinner("Analyzing voice patterns..."):
                try:
                    features = voice_proc.extract_features(audio_file)
                    
                    if features is not None:
                        feature_array = np.array([features[name] for name in voice_proc.feature_names]).reshape(1, -1)
                        
                        prediction = model_loader.predict_voice(feature_array, model_type='mlp')
                        confidence = prediction['confidence']
                        has_pd = prediction['prediction'] == 1
                        
                        st.divider()
                        
                        if has_pd:
                            st.error("### ⚠️ PARKINSON'S DISEASE INDICATORS DETECTED")
                            st.markdown(f"**Confidence Level:** {confidence:.1f}%")
                            
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
                                   
                                3. **Physical Therapist**
                                   - Early intervention for motor symptoms
                                   
                                4. **Speech-Language Pathologist**
                                   - For voice and communication support
                                """)
                            
                            with col2:
                                st.markdown("#### 📋 Tests Your Doctor May Order")
                                st.markdown("""
                                - **DaTscan** - Brain imaging for dopamine levels
                                - **MRI Brain Scan** - Rule out other conditions
                                - **Blood Tests** - Exclude other causes
                                - **Complete Neurological Exam** - Motor function assessment
                                - **Genetic Testing** - If family history present
                                - **Smell Test** - Loss of smell is early indicator
                                """)
                            
                            st.divider()
                            
                            st.markdown("### 🛡️ Immediate Precautions & Lifestyle Modifications")
                            
                            tab1, tab2, tab3, tab4 = st.tabs(["🏃 Physical Care", "🍎 Nutrition", "🧘 Mental Health", "⚠️ Safety"])
                            
                            with tab1:
                                st.markdown("""
                                #### Daily Exercise Routine (30-60 minutes)
                                - **Walking/Aerobic Exercise**: Improves balance and cardiovascular health
                                - **Stretching**: Reduces muscle stiffness and rigidity
                                - **Strength Training**: 2-3 times per week to maintain muscle mass
                                - **Tai Chi or Yoga**: Enhances balance and flexibility
                                - **Dancing**: Excellent for coordination and mood
                                
                                #### Physical Therapy Goals
                                - Maintain range of motion
                                - Improve posture and gait
                                - Prevent falls
                                - Address freezing episodes
                                """)
                            
                            with tab2:
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
                            
                            with tab3:
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
                            
                            with tab4:
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
                                
                                #### Medication Safety
                                - Keep detailed medication log
                                - Set alarms for medication times
                                - Store medications properly
                                - Regular medication reviews with doctor
                                """)
                            
                            st.divider()
                            
                            st.markdown("### 📊 Voice Analysis Details")
                            with st.expander("Click to view detailed voice metrics"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Jitter", f"{features.get('Jitter_percent', 0):.4f}%", 
                                             "High" if features.get('Jitter_percent', 0) > 0.006 else "Normal")
                                    st.metric("Shimmer", f"{features.get('Shimmer', 0):.4f}", 
                                             "High" if features.get('Shimmer', 0) > 0.03 else "Normal")
                                with col2:
                                    st.metric("HNR", f"{features.get('HNR', 0):.2f} dB", 
                                             "Low" if features.get('HNR', 0) < 20 else "Normal")
                                    st.metric("Fundamental Freq", f"{features.get('MDVP_Fo_Hz', 0):.1f} Hz")
                            
                            st.divider()
                            
                            st.success("### 💚 Important Reminder")
                            st.markdown("""
                            **Early detection and intervention can significantly improve quality of life!**
                            
                            - Parkinson's is manageable with proper treatment
                            - Many people live full, active lives with PD
                            - Treatments are continuously improving
                            - Early intervention provides best outcomes
                            
                            **Next Steps:**
                            1. Print or save this report
                            2. Schedule neurologist appointment within 1-2 weeks
                            3. Start implementing lifestyle modifications
                            4. Keep a symptom diary to share with your doctor
                            5. Bring a family member to your appointment
                            """)
                        else:
                            st.success("### ✅ NO PARKINSON'S DISEASE DETECTED")
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            st.markdown("**Result:** Voice patterns appear normal.")
                            
                            with st.expander("📊 Voice Analysis Details"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Jitter", f"{features.get('Jitter_percent', 0):.4f}%", "Normal")
                                    st.metric("Shimmer", f"{features.get('Shimmer', 0):.4f}", "Normal")
                                with col2:
                                    st.metric("HNR", f"{features.get('HNR', 0):.2f} dB", "Normal")
                                    st.metric("Fundamental Freq", f"{features.get('MDVP_Fo_Hz', 0):.1f} Hz")
                    else:
                        st.error("Could not analyze the audio file. Please try another file.")
                        
                except Exception as e:
                    st.error(f"Error analyzing audio: {str(e)}")

with tab2:
    st.markdown("#### Upload a handwriting sample")
    st.info("📋 Upload a spiral or wave drawing image (PNG, JPG)")
    
    image_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'], key='image')
    
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)
        
        if st.button("🔍 Analyze Handwriting", type="primary", use_container_width=True):
            with st.spinner("Analyzing handwriting patterns..."):
                try:
                    img_array = np.array(image)
                    features = handwriting_proc.extract_features(img_array)
                    
                    if features is not None:
                        feature_array = np.array([features[name] for name in handwriting_proc.feature_names]).reshape(1, -1)
                        
                        prediction = model_loader.predict_handwriting(feature_array, model_type='cnn')
                        confidence = prediction['confidence']
                        has_pd = prediction['prediction'] == 1
                        
                        st.divider()
                        
                        if has_pd:
                            st.error("### ⚠️ PARKINSON'S DISEASE INDICATORS DETECTED")
                            st.markdown(f"**Confidence Level:** {confidence:.1f}%")
                            
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
                            
                            tab1, tab2, tab3, tab4 = st.tabs(["🏃 Physical Care", "🍎 Nutrition", "🧘 Mental Health", "⚠️ Safety"])
                            
                            with tab1:
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
                            
                            with tab2:
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
                            
                            with tab3:
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
                            
                            with tab4:
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
                            
                            st.markdown("### 📊 Handwriting Analysis Details")
                            with st.expander("Click to view detailed handwriting metrics"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Tremor Score", f"{features.get('tremor_score', 0):.2f}", 
                                             "High" if features.get('tremor_score', 0) > 0.5 else "Normal")
                                    st.metric("Pressure Variation", f"{features.get('pressure_variation', 0):.2f}", 
                                             "High" if features.get('pressure_variation', 0) > 0.4 else "Normal")
                                with col2:
                                    st.metric("Line Deviation", f"{features.get('line_deviation', 0):.2f}", 
                                             "High" if features.get('line_deviation', 0) > 0.3 else "Normal")
                                    st.metric("Stroke Smoothness", f"{features.get('stroke_smoothness', 0):.2f}")
                            
                            st.divider()
                            
                            st.success("### 💚 Important Reminder")
                            st.markdown("""
                            **Early detection and intervention can significantly improve quality of life!**
                            
                            - Parkinson's is manageable with proper treatment
                            - Many people live full, active lives with PD
                            - Treatments are continuously improving
                            - Early intervention provides best outcomes
                            
                            **Next Steps:**
                            1. Print or save this report
                            2. Schedule neurologist appointment within 1-2 weeks
                            3. Start implementing lifestyle modifications
                            4. Keep a symptom diary to share with your doctor
                            5. Bring a family member to your appointment
                            """)
                        else:
                            st.success("### ✅ NO PARKINSON'S DISEASE DETECTED")
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            st.markdown("**Result:** Handwriting patterns appear normal.")
                            
                            with st.expander("📊 Handwriting Analysis Details"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Tremor Score", f"{features.get('tremor_score', 0):.2f}", "Normal")
                                    st.metric("Pressure Variation", f"{features.get('pressure_variation', 0):.2f}", "Normal")
                                with col2:
                                    st.metric("Line Deviation", f"{features.get('line_deviation', 0):.2f}", "Normal")
                                    st.metric("Stroke Smoothness", f"{features.get('stroke_smoothness', 0):.2f}")
                    else:
                        st.error("Could not analyze the image. Please try another file.")
                        
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")

st.divider()
st.markdown("""
### 📥 Download Sample Files for Testing

**Voice Samples:**
- `parkinsons_voice_sample.wav` - Voice with PD indicators
- `healthy_voice_sample.wav` - Normal voice

**Handwriting Samples:**
- `parkinsons_spiral.png` - Spiral with tremor (PD positive)
- `healthy_spiral.png` - Normal spiral drawing
- `parkinsons_wave.png` - Wave with tremor (PD positive)
- `healthy_wave.png` - Normal wave drawing

All files are available in the project root folder for download.
""")

st.info("⚠️ **Disclaimer:** This is an AI-based screening tool for educational purposes. It does not replace professional medical diagnosis. Always consult a healthcare professional for proper evaluation.")
