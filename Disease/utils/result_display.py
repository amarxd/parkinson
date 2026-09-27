"""
Unified result display module for Parkinson's Disease detection results.
Provides consistent PD positive/negative status display across all analysis pages.
"""

import streamlit as st
import plotly.graph_objects as go


def display_pd_result(prediction, confidence, analysis_type="Analysis"):
    """
    Display unified PD detection result with color-coded status.
    
    Args:
        prediction: 1 for PD positive, 0 for healthy
        confidence: Confidence score (0-1)
        analysis_type: Type of analysis (Voice, Handwriting, Facial, Motion)
    """
    
    # Determine result status
    if prediction == 1:
        status = "⚠️ PARKINSON'S DISEASE INDICATORS DETECTED"
        status_color = "#ff4444"
        bg_color = "#ffe6e6"
        recommendation = "**RECOMMENDATION**: Consult with a neurologist for comprehensive evaluation."
    else:
        status = "✅ NO SIGNIFICANT PARKINSON'S INDICATORS DETECTED"
        status_color = "#00cc66"
        bg_color = "#e6f7f0"
        recommendation = "**RECOMMENDATION**: Continue regular health monitoring."
    
    # Display result banner
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border-left: 5px solid {status_color};
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    ">
        <h2 style="color: {status_color}; margin: 0 0 10px 0;">{status}</h2>
        <p style="font-size: 18px; margin: 10px 0;">
            <strong>{analysis_type} Confidence:</strong> {confidence*100:.1f}%
        </p>
        <p style="margin: 10px 0;">
            {recommendation}
        </p>
        <p style="font-size: 14px; color: #666; margin: 10px 0 0 0;">
            <em>⚕️ IMPORTANT: This is a screening tool only. Clinical diagnosis requires professional medical evaluation.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display confidence meter
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"{analysis_type} Detection Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': status_color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightcoral" if prediction == 1 else "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_combined_pd_result(results_dict):
    """
    Display combined PD result from multiple analysis types.
    
    Args:
        results_dict: Dictionary with keys as analysis types and values as (prediction, confidence) tuples
                     Example: {'Voice': (1, 0.87), 'Handwriting': (1, 0.92)}
    """
    if not results_dict:
        st.warning("No analysis results available.")
        return
    
    # Calculate overall assessment
    total_predictions = sum(pred for pred, _ in results_dict.values())
    avg_confidence = sum(conf for _, conf in results_dict.values()) / len(results_dict)
    
    # Determine overall status
    pd_detected_count = total_predictions
    total_tests = len(results_dict)
    
    if pd_detected_count >= (total_tests / 2):
        overall_status = "⚠️ PARKINSON'S DISEASE INDICATORS DETECTED"
        overall_color = "#ff4444"
        overall_bg = "#ffe6e6"
        recommendation = "**URGENT**: Schedule appointment with neurologist for comprehensive evaluation."
    else:
        overall_status = "✅ NO SIGNIFICANT PARKINSON'S INDICATORS"
        overall_color = "#00cc66"
        overall_bg = "#e6f7f0"
        recommendation = "**RECOMMENDATION**: Results are within normal range. Continue regular health monitoring."
    
    # Display overall result
    st.markdown(f"""
    <div style="
        background-color: {overall_bg};
        border-left: 8px solid {overall_color};
        padding: 25px;
        border-radius: 8px;
        margin: 25px 0;
    ">
        <h1 style="color: {overall_color}; margin: 0 0 15px 0;">{overall_status}</h1>
        <p style="font-size: 20px; margin: 15px 0;">
            <strong>Overall Confidence:</strong> {avg_confidence*100:.1f}%
        </p>
        <p style="font-size: 18px; margin: 15px 0;">
            <strong>Tests with PD Indicators:</strong> {pd_detected_count} out of {total_tests}
        </p>
        <p style="margin: 15px 0;">
            {recommendation}
        </p>
        <p style="font-size: 14px; color: #666; margin: 15px 0 0 0;">
            <em>⚕️ MEDICAL DISCLAIMER: This AI-powered screening tool is for educational and preliminary assessment purposes only. 
            It does NOT replace professional medical diagnosis. Always consult qualified healthcare providers for clinical evaluation.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display individual test results
    st.markdown("### 📊 Individual Test Results")
    cols = st.columns(len(results_dict))
    
    for idx, (test_name, (prediction, confidence)) in enumerate(results_dict.items()):
        with cols[idx]:
            status_icon = "⚠️" if prediction == 1 else "✅"
            status_text = "PD Detected" if prediction == 1 else "Healthy"
            color = "#ff4444" if prediction == 1 else "#00cc66"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 48px;">{status_icon}</div>
                <h3>{test_name}</h3>
                <p style="color: {color}; font-weight: bold;">{status_text}</p>
                <p>Confidence: {confidence*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)


def display_medical_disclaimer():
    """Display medical disclaimer and next steps."""
    st.markdown("""
    ---
    ### ⚕️ Important Medical Information
    
    **Clinical Consultation Required:**
    - This screening tool is NOT a diagnostic device
    - Parkinson's Disease diagnosis requires comprehensive neurological examination
    - Multiple factors must be evaluated by qualified healthcare professionals
    
    **Next Steps if PD Indicators Detected:**
    1. **Schedule neurologist appointment** - Seek specialist evaluation
    2. **Document symptoms** - Keep detailed records of observations
    3. **Family history** - Prepare information about family medical history
    4. **Current medications** - List all current medications and supplements
    5. **Lifestyle factors** - Note any relevant lifestyle or environmental factors
    
    **Additional Testing May Include:**
    - Complete neurological examination
    - DaTscan or other imaging studies
    - Blood tests to rule out other conditions
    - Physical and occupational therapy assessments
    - Genetic testing (if indicated)
    
    **Resources:**
    - Parkinson's Foundation: parkinson.org
    - American Parkinson Disease Association: apdaparkinson.org
    - Michael J. Fox Foundation: michaeljfox.org
    """)
