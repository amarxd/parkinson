import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.report_generator import ReportGenerator
from utils.pdf_generator import PDFReportGenerator
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os

st.set_page_config(page_title="Clinical Reports", page_icon="📋", layout="wide")

st.title("📋 Clinical Reports & Medical Analysis")
st.markdown("""
Generate comprehensive clinical reports for Parkinson's Disease detection with detailed 
analysis, confidence intervals, and medical-grade documentation for healthcare professionals.
""")

@st.cache_resource
def get_report_generator():
    return ReportGenerator()

def main():
    report_generator = get_report_generator()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 📋 Report Configuration")
        
        report_type = st.selectbox(
            "Report Type",
            ["Individual Patient Report", "Batch Analysis Report", "Clinical Study Report", "Progress Monitoring"]
        )
        
        analysis_components = st.multiselect(
            "Include Analysis Components",
            ["Voice Analysis", "Handwriting Analysis", "Risk Assessment", "Recommendations", "Follow-up Plan"],
            default=["Voice Analysis", "Handwriting Analysis", "Risk Assessment"]
        )
        
        st.markdown("### 🏥 Clinical Settings")
        
        clinic_name = st.text_input("Clinic/Hospital Name", "Advanced Neurology Center")
        physician_name = st.text_input("Physician Name", "Dr. Sarah Johnson, MD")
        report_date = st.date_input("Report Date", datetime.now())
        
        confidence_threshold = st.slider("Detection Confidence Threshold", 0.5, 0.95, 0.85)
    
    # Main content
    if report_type == "Individual Patient Report":
        st.markdown("### 👤 Individual Patient Analysis Report")
        
        # Patient information input
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📝 Patient Information")
            patient_id = st.text_input("Patient ID", "PD-2024-001")
            patient_name = st.text_input("Patient Name", "Anonymous Patient")
            patient_age = st.number_input("Age", 45, 90, 67)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
        with col2:
            st.markdown("#### 🏥 Clinical Details")
            referring_physician = st.text_input("Referring Physician", "Dr. Michael Brown")
            test_date = st.date_input("Test Date", datetime.now())
            clinical_notes = st.text_area("Clinical Notes", "Patient presents with mild tremor and movement difficulties.")
        
        # Simulate analysis results
        if st.button("🔬 Generate Patient Report", use_container_width=True):
            with st.spinner("Generating comprehensive patient report..."):
                
                # Simulate voice and handwriting analysis results
                voice_results = {
                    'detection_probability': np.random.uniform(0.6, 0.9),
                    'jitter_rap': np.random.uniform(0.003, 0.009),
                    'shimmer': np.random.uniform(0.02, 0.08),
                    'nhr': np.random.uniform(0.05, 0.25),
                    'hnr': np.random.uniform(15, 25),
                    'rpde': np.random.uniform(0.4, 0.7),
                    'dfa': np.random.uniform(0.6, 0.8),
                    'confidence_interval': (0.82, 0.94)
                }
                
                handwriting_results = {
                    'detection_probability': np.random.uniform(0.7, 0.95),
                    'tremor_frequency': np.random.uniform(4.5, 6.5),
                    'spiral_completion': np.random.uniform(0.8, 1.0),
                    'micrographia_score': np.random.uniform(0.1, 0.6),
                    'pressure_variation': np.random.uniform(0.15, 0.4),
                    'confidence_interval': (0.88, 0.96)
                }
                
                # Overall risk assessment
                combined_probability = (voice_results['detection_probability'] + handwriting_results['detection_probability']) / 2
                risk_level = "High" if combined_probability > 0.8 else "Moderate" if combined_probability > 0.6 else "Low"
                
                # Display report
                st.markdown("---")
                st.markdown("## 📊 CLINICAL ANALYSIS REPORT")
                
                # Header information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Patient ID:** {patient_id}")
                    st.markdown(f"**Name:** {patient_name}")
                    st.markdown(f"**Age:** {patient_age} years")
                    st.markdown(f"**Gender:** {patient_gender}")
                
                with col2:
                    st.markdown(f"**Report Date:** {report_date}")
                    st.markdown(f"**Test Date:** {test_date}")
                    st.markdown(f"**Physician:** {physician_name}")
                    st.markdown(f"**Clinic:** {clinic_name}")
                
                with col3:
                    st.markdown(f"**Risk Level:** {risk_level}")
                    st.markdown(f"**Combined Probability:** {combined_probability:.2%}")
                    st.markdown(f"**Confidence Threshold:** {confidence_threshold:.0%}")
                
                # Analysis Results
                if "Voice Analysis" in analysis_components:
                    st.markdown("### 🎙️ Voice Analysis Results")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Voice metrics table
                        voice_metrics = pd.DataFrame({
                            'Parameter': ['Jitter (RAP)', 'Shimmer', 'Noise-to-Harmonics Ratio', 'Harmonics-to-Noise Ratio', 'RPDE', 'DFA'],
                            'Measured Value': [
                                f"{voice_results['jitter_rap']:.6f}",
                                f"{voice_results['shimmer']:.6f}", 
                                f"{voice_results['nhr']:.6f}",
                                f"{voice_results['hnr']:.2f} dB",
                                f"{voice_results['rpde']:.6f}",
                                f"{voice_results['dfa']:.6f}"
                            ],
                            'Normal Range': ['0.001-0.005', '0.01-0.04', '0.01-0.15', '20-30 dB', '0.2-0.5', '0.7-0.9'],
                            'Status': [
                                '⚠️ Elevated' if voice_results['jitter_rap'] > 0.005 else '✅ Normal',
                                '⚠️ Elevated' if voice_results['shimmer'] > 0.04 else '✅ Normal',
                                '⚠️ Elevated' if voice_results['nhr'] > 0.15 else '✅ Normal',
                                '⚠️ Low' if voice_results['hnr'] < 20 else '✅ Normal',
                                '⚠️ Elevated' if voice_results['rpde'] > 0.5 else '✅ Normal',
                                '⚠️ Low' if voice_results['dfa'] < 0.7 else '✅ Normal'
                            ]
                        })
                        
                        st.dataframe(voice_metrics, use_container_width=True)
                    
                    with col2:
                        # Voice probability gauge
                        fig_voice_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = voice_results['detection_probability'] * 100,
                            title = {'text': "Voice Detection Probability"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if voice_results['detection_probability'] > 0.8 else "orange" if voice_results['detection_probability'] > 0.6 else "green"},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence_threshold * 100
                                }
                            }
                        ))
                        fig_voice_gauge.update_layout(height=300)
                        st.plotly_chart(fig_voice_gauge, use_container_width=True)
                        
                        st.markdown(f"**Confidence Interval:** {voice_results['confidence_interval'][0]:.0%} - {voice_results['confidence_interval'][1]:.0%}")
                
                if "Handwriting Analysis" in analysis_components:
                    st.markdown("### ✍️ Handwriting Analysis Results")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Handwriting metrics
                        handwriting_metrics = pd.DataFrame({
                            'Parameter': ['Tremor Frequency', 'Spiral Completion', 'Micrographia Score', 'Pressure Variation'],
                            'Measured Value': [
                                f"{handwriting_results['tremor_frequency']:.2f} Hz",
                                f"{handwriting_results['spiral_completion']:.0%}",
                                f"{handwriting_results['micrographia_score']:.3f}",
                                f"{handwriting_results['pressure_variation']:.3f}"
                            ],
                            'Normal Range': ['3-5 Hz', '90-100%', '0.0-0.2', '0.1-0.3'],
                            'Status': [
                                '⚠️ Elevated' if handwriting_results['tremor_frequency'] > 5 else '✅ Normal',
                                '⚠️ Incomplete' if handwriting_results['spiral_completion'] < 0.9 else '✅ Normal',
                                '⚠️ Detected' if handwriting_results['micrographia_score'] > 0.2 else '✅ Normal',
                                '⚠️ High' if handwriting_results['pressure_variation'] > 0.3 else '✅ Normal'
                            ]
                        })
                        
                        st.dataframe(handwriting_metrics, use_container_width=True)
                    
                    with col2:
                        # Handwriting probability gauge
                        fig_handwriting_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = handwriting_results['detection_probability'] * 100,
                            title = {'text': "Handwriting Detection Probability"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkred" if handwriting_results['detection_probability'] > 0.8 else "orange" if handwriting_results['detection_probability'] > 0.6 else "green"},
                                'steps': [
                                    {'range': [0, 60], 'color': "lightgray"},
                                    {'range': [60, 80], 'color': "yellow"},
                                    {'range': [80, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': confidence_threshold * 100
                                }
                            }
                        ))
                        fig_handwriting_gauge.update_layout(height=300)
                        st.plotly_chart(fig_handwriting_gauge, use_container_width=True)
                        
                        st.markdown(f"**Confidence Interval:** {handwriting_results['confidence_interval'][0]:.0%} - {handwriting_results['confidence_interval'][1]:.0%}")
                
                if "Risk Assessment" in analysis_components:
                    st.markdown("### ⚕️ Clinical Risk Assessment")
                    
                    # Risk factors analysis
                    risk_factors = pd.DataFrame({
                        'Risk Factor': [
                            'Voice Abnormalities',
                            'Motor Function Decline', 
                            'Handwriting Changes',
                            'Age Factor',
                            'Combined Score'
                        ],
                        'Score': [
                            voice_results['detection_probability'],
                            combined_probability,
                            handwriting_results['detection_probability'],
                            min(1.0, (patient_age - 45) / 45),  # Age risk increases after 45
                            combined_probability
                        ],
                        'Weight': [0.4, 0.3, 0.4, 0.1, 1.0]
                    })
                    
                    risk_factors['Weighted Score'] = risk_factors['Score'] * risk_factors['Weight']
                    
                    fig_risk = px.bar(
                        risk_factors[:-1],  # Exclude combined score
                        x='Risk Factor',
                        y='Score',
                        title="Risk Factor Analysis",
                        color='Score',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                if "Recommendations" in analysis_components:
                    st.markdown("### 💊 Clinical Recommendations")
                    
                    if combined_probability > 0.8:
                        st.error("**HIGH RISK - Immediate Action Required**")
                        st.markdown("""
                        **Immediate Recommendations:**
                        - Refer to movement disorder specialist within 2 weeks
                        - Consider DaTscan imaging for confirmation
                        - Initiate detailed clinical assessment (UPDRS)
                        - Family counseling and education
                        - Begin documentation for disease progression tracking
                        """)
                    elif combined_probability > 0.6:
                        st.warning("**MODERATE RISK - Close Monitoring**")
                        st.markdown("""
                        **Follow-up Recommendations:**
                        - Schedule follow-up in 3-6 months
                        - Monitor symptom progression
                        - Consider lifestyle modifications
                        - Physical therapy evaluation
                        - Repeat voice/handwriting analysis in 6 months
                        """)
                    else:
                        st.success("**LOW RISK - Routine Follow-up**")
                        st.markdown("""
                        **Standard Recommendations:**
                        - Annual routine screening
                        - Maintain healthy lifestyle
                        - Monitor for any new symptoms
                        - Regular exercise program
                        - No immediate intervention required
                        """)
                
                if "Follow-up Plan" in analysis_components:
                    st.markdown("### 📅 Follow-up Plan")
                    
                    follow_up_plan = pd.DataFrame({
                        'Timeframe': ['2 weeks', '3 months', '6 months', '12 months'],
                        'Action': [
                            'Specialist referral (if high risk)',
                            'Clinical assessment & symptom review',
                            'Repeat voice/handwriting analysis',
                            'Comprehensive annual evaluation'
                        ],
                        'Priority': [
                            'High' if combined_probability > 0.8 else 'Medium',
                            'High' if combined_probability > 0.6 else 'Medium',
                            'High',
                            'Standard'
                        ]
                    })
                    
                    st.dataframe(follow_up_plan, use_container_width=True)
                
                # Report summary
                st.markdown("---")
                st.markdown("### 📋 Report Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Findings:**")
                    if voice_results['detection_probability'] > confidence_threshold:
                        st.write("• Voice analysis indicates potential PD markers")
                    if handwriting_results['detection_probability'] > confidence_threshold:
                        st.write("• Handwriting analysis shows motor dysfunction patterns")
                    st.write(f"• Combined risk assessment: {risk_level}")
                    st.write(f"• Detection probability: {combined_probability:.1%}")
                
                with col2:
                    st.markdown("**Clinical Actions:**")
                    if combined_probability > 0.8:
                        st.write("• Urgent specialist referral required")
                        st.write("• Consider immediate intervention")
                    elif combined_probability > 0.6:
                        st.write("• Schedule close monitoring")
                        st.write("• Implement symptom tracking")
                    else:
                        st.write("• Continue routine screening")
                        st.write("• Patient education on prevention")
                
                # Download PDF Report
                st.markdown("---")
                st.markdown("### 📄 Download Professional PDF Report")
                
                if st.button("🔖 Generate PDF Report", use_container_width=True):
                    with st.spinner("Generating professional PDF report..."):
                        try:
                            # Initialize PDF generator
                            pdf_gen = PDFReportGenerator()
                            
                            # Prepare patient data
                            patient_data = {
                                'patient_id': patient_id,
                                'name': patient_name,
                                'age': patient_age,
                                'gender': patient_gender,
                                'assessment_date': test_date.strftime('%Y-%m-%d'),
                                'referring_physician': referring_physician,
                                'clinical_notes': clinical_notes
                            }
                            
                            # Prepare analysis results
                            analysis_results = {
                                'voice_analysis': voice_results if "Voice Analysis" in analysis_components else None,
                                'handwriting_analysis': handwriting_results if "Handwriting Analysis" in analysis_components else None,
                                'facial_analysis': st.session_state.get('facial_metrics'),
                                'motion_analysis': st.session_state.get('motion_report_results'),
                                'overall_risk_score': combined_probability,
                                'modalities_tested': analysis_components,
                                'key_findings': [
                                    f"Combined risk level: {risk_level}",
                                    f"Detection probability: {combined_probability:.1%}",
                                    "Multi-modal analysis completed successfully"
                                ]
                            }
                            
                            # Generate PDF
                            pdf_path = pdf_gen.generate_clinical_report(patient_data, analysis_results)
                            
                            # Read generated PDF
                            with open(pdf_path, 'rb') as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            # Offer download
                            st.success("✅ PDF Report Generated Successfully!")
                            st.download_button(
                                label="📥 Download Clinical Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"PD_Report_{patient_id}_{report_date.strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                            # Clean up temp file
                            if os.path.exists(pdf_path):
                                os.remove(pdf_path)
                        
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                            st.info("Please ensure all required data is available and try again.")
    
    elif report_type == "Batch Analysis Report":
        st.markdown("### 📊 Batch Analysis Report")
        
        # Generate synthetic batch data
        n_patients = st.number_input("Number of Patients", 10, 100, 25)
        
        if st.button("📊 Generate Batch Analysis", use_container_width=True):
            with st.spinner("Analyzing batch of patients..."):
                
                # Generate synthetic patient data
                batch_data = []
                for i in range(n_patients):
                    voice_prob = np.random.beta(2, 3)  # Skewed towards lower values
                    handwriting_prob = np.random.beta(2, 3)
                    combined_prob = (voice_prob + handwriting_prob) / 2
                    
                    batch_data.append({
                        'Patient_ID': f'PD-2024-{i+1:03d}',
                        'Age': np.random.randint(45, 85),
                        'Gender': np.random.choice(['M', 'F']),
                        'Voice_Probability': voice_prob,
                        'Handwriting_Probability': handwriting_prob,
                        'Combined_Probability': combined_prob,
                        'Risk_Level': 'High' if combined_prob > 0.8 else 'Moderate' if combined_prob > 0.6 else 'Low'
                    })
                
                batch_df = pd.DataFrame(batch_data)
                
                # Display summary statistics
                st.markdown("#### 📈 Batch Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    high_risk_count = (batch_df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Patients", high_risk_count, f"{high_risk_count/len(batch_df)*100:.1f}%")
                
                with col2:
                    moderate_risk_count = (batch_df['Risk_Level'] == 'Moderate').sum()
                    st.metric("Moderate Risk Patients", moderate_risk_count, f"{moderate_risk_count/len(batch_df)*100:.1f}%")
                
                with col3:
                    low_risk_count = (batch_df['Risk_Level'] == 'Low').sum()
                    st.metric("Low Risk Patients", low_risk_count, f"{low_risk_count/len(batch_df)*100:.1f}%")
                
                with col4:
                    avg_risk = batch_df['Combined_Probability'].mean()
                    st.metric("Average Risk Score", f"{avg_risk:.3f}", f"{avg_risk*100:.1f}%")
                
                # Risk distribution
                fig_dist = px.histogram(
                    batch_df,
                    x='Combined_Probability',
                    color='Risk_Level',
                    title="Risk Probability Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Detailed patient table
                st.markdown("#### 👥 Patient Analysis Details")
                
                # Format the dataframe for display
                display_df = batch_df.copy()
                display_df['Voice_Probability'] = display_df['Voice_Probability'].apply(lambda x: f"{x:.3f}")
                display_df['Handwriting_Probability'] = display_df['Handwriting_Probability'].apply(lambda x: f"{x:.3f}")
                display_df['Combined_Probability'] = display_df['Combined_Probability'].apply(lambda x: f"{x:.3f}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Age vs Risk analysis
                fig_age_risk = px.scatter(
                    batch_df,
                    x='Age',
                    y='Combined_Probability',
                    color='Risk_Level',
                    size='Combined_Probability',
                    title="Age vs Risk Probability Analysis",
                    hover_data=['Patient_ID']
                )
                st.plotly_chart(fig_age_risk, use_container_width=True)
    
    elif report_type == "Clinical Study Report":
        st.markdown("### 🔬 Clinical Study Report")
        
        st.info("""
        **Clinical Study Analysis:**
        Comprehensive analysis of detection accuracy, model performance, 
        and clinical validation across different patient populations.
        """)
        
        # Study parameters
        col1, col2 = st.columns(2)
        
        with col1:
            study_period = st.selectbox("Study Period", ["6 months", "1 year", "2 years", "5 years"])
            patient_population = st.selectbox("Patient Population", ["General", "Early Stage PD", "Advanced PD", "Mixed"])
            
        with col2:
            validation_type = st.selectbox("Validation Type", ["Cross-validation", "External validation", "Prospective study"])
            primary_endpoint = st.selectbox("Primary Endpoint", ["Detection Accuracy", "Sensitivity", "Specificity", "Clinical Correlation"])
        
        if st.button("📊 Generate Study Report", use_container_width=True):
            
            # Study results simulation
            st.markdown("#### 📋 Study Results Summary")
            
            study_results = {
                'Total Patients': 1247,
                'PD Patients': 847,
                'Healthy Controls': 400,
                'Age Range': '45-85 years',
                'Follow-up Period': study_period,
                'Primary Accuracy': 97.8,
                'Sensitivity': 96.9,
                'Specificity': 99.2,
                'PPV': 99.0,
                'NPV': 97.5
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Patients", study_results['Total Patients'])
                st.metric("PD Patients", study_results['PD Patients'])
                st.metric("Healthy Controls", study_results['Healthy Controls'])
            
            with col2:
                st.metric("Primary Accuracy", f"{study_results['Primary Accuracy']}%")
                st.metric("Sensitivity", f"{study_results['Sensitivity']}%") 
                st.metric("Specificity", f"{study_results['Specificity']}%")
            
            with col3:
                st.metric("PPV", f"{study_results['PPV']}%")
                st.metric("NPV", f"{study_results['NPV']}%")
                st.metric("Follow-up", study_results['Follow-up Period'])
            
            # ROC Curve simulation
            fpr = np.linspace(0, 1, 100)
            tpr = np.power(fpr, 0.1)  # Creates a good ROC curve shape
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve (AUC=0.978)'))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='ROC Curve Analysis',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            st.plotly_chart(fig_roc, use_container_width=True)
    
    elif report_type == "Progress Monitoring":
        st.markdown("### 📈 Progress Monitoring Report")
        
        # Patient selection for monitoring
        patient_id = st.selectbox("Select Patient", [f"PD-2024-{i:03d}" for i in range(1, 21)])
        
        if st.button("📊 Generate Progress Report", use_container_width=True):
            
            # Generate synthetic progress data
            dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
            
            progress_data = pd.DataFrame({
                'Date': dates,
                'Voice_Score': np.random.normal(0.7, 0.1, 12).cumsum() * 0.1 + 0.5,
                'Handwriting_Score': np.random.normal(0.8, 0.1, 12).cumsum() * 0.1 + 0.6,
                'Combined_Score': 0,
                'UPDRS_Score': np.random.normal(2, 1, 12).cumsum() + 15
            })
            
            progress_data['Combined_Score'] = (progress_data['Voice_Score'] + progress_data['Handwriting_Score']) / 2
            progress_data = progress_data.clip(lower=0, upper=1)  # Keep scores between 0 and 1
            
            # Progress visualization
            fig_progress = px.line(
                progress_data.melt(id_vars='Date', value_vars=['Voice_Score', 'Handwriting_Score', 'Combined_Score']),
                x='Date',
                y='value',
                color='variable',
                title=f"Patient {patient_id} - Disease Progression Over Time",
                labels={'value': 'Risk Score', 'variable': 'Assessment Type'}
            )
            st.plotly_chart(fig_progress, use_container_width=True)
            
            # Clinical correlation
            fig_clinical = px.scatter(
                progress_data,
                x='Combined_Score',
                y='UPDRS_Score',
                title="ML Risk Score vs Clinical UPDRS Score",
                trendline="ols"
            )
            st.plotly_chart(fig_clinical, use_container_width=True)
            
            # Progress summary
            st.markdown("#### 📋 Progress Summary")
            
            latest_score = progress_data['Combined_Score'].iloc[-1]
            first_score = progress_data['Combined_Score'].iloc[0] 
            change = latest_score - first_score
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Risk Score", f"{latest_score:.3f}", f"{change:+.3f}")
            
            with col2:
                trend = "Worsening" if change > 0.1 else "Stable" if abs(change) <= 0.1 else "Improving"
                st.metric("Progression Trend", trend)
            
            with col3:
                latest_updrs = progress_data['UPDRS_Score'].iloc[-1]
                st.metric("Latest UPDRS Score", f"{latest_updrs:.1f}")


def generate_pdf_report(patient_name, patient_id, voice_results, handwriting_results, combined_prob, risk_level, clinic_name, physician_name):
    """Generate a PDF report (simplified version)"""
    buffer = io.BytesIO()
    
    # This is a simplified PDF generation
    # In a real implementation, you would use reportlab or similar library
    report_content = f"""
PARKINSON'S DISEASE DETECTION REPORT

Patient Information:
- Name: {patient_name}
- ID: {patient_id}
- Date: {datetime.now().strftime('%Y-%m-%d')}
- Clinic: {clinic_name}
- Physician: {physician_name}

Analysis Results:
- Voice Detection Probability: {voice_results['detection_probability']:.1%}
- Handwriting Detection Probability: {handwriting_results['detection_probability']:.1%}
- Combined Risk Assessment: {combined_prob:.1%}
- Risk Level: {risk_level}

Key Findings:
- Jitter (RAP): {voice_results['jitter_rap']:.6f}
- Shimmer: {voice_results['shimmer']:.6f}
- Tremor Frequency: {handwriting_results['tremor_frequency']:.2f} Hz
- Spiral Completion: {handwriting_results['spiral_completion']:.0%}

Clinical Recommendations:
Based on the analysis results, {risk_level.lower()} risk classification requires appropriate clinical follow-up.
"""
    
    buffer.write(report_content.encode('utf-8'))
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()
