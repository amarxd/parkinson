from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from datetime import datetime
import io
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class PDFReportGenerator:
    """
    Generate professional PDF clinical reports for Parkinson's Disease detection analysis.
    """
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section heading style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=12,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#e74c3c'),
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        
        # Info style
        self.styles.add(ParagraphStyle(
            name='Info',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#3498db'),
            spaceAfter=6
        ))
    
    def generate_clinical_report(self, patient_data, analysis_results, output_path=None):
        """
        Generate a comprehensive clinical PDF report.
        
        Args:
            patient_data: Dictionary containing patient information
            analysis_results: Dictionary containing all analysis results
            output_path: Path to save PDF (default: temp file)
            
        Returns:
            str: Path to generated PDF file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/parkinsons_report_{timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build document content
        story = []
        
        # Add header
        story.extend(self._create_header())
        
        # Add patient information
        story.extend(self._create_patient_section(patient_data))
        
        # Add executive summary
        story.extend(self._create_executive_summary(analysis_results))
        
        # Add detailed analysis sections
        if analysis_results.get('voice_analysis'):
            story.extend(self._create_voice_analysis_section(analysis_results['voice_analysis']))
        
        if analysis_results.get('handwriting_analysis'):
            story.extend(self._create_handwriting_section(analysis_results['handwriting_analysis']))
        
        if analysis_results.get('facial_analysis'):
            story.extend(self._create_facial_section(analysis_results['facial_analysis']))
        
        if analysis_results.get('motion_analysis'):
            story.extend(self._create_motion_section(analysis_results['motion_analysis']))
        
        # Add clinical recommendations
        story.extend(self._create_recommendations_section(analysis_results))
        
        # Add disclaimer
        story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def _create_header(self):
        """Create report header."""
        elements = []
        
        # Title
        title = Paragraph("Parkinson's Disease Detection Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y %H:%M")
        metadata = Paragraph(
            f"<b>Report Generated:</b> {report_date}<br/>"
            f"<b>System Version:</b> Advanced Multi-Modal AI Detection System v2.0<br/>"
            f"<b>Analysis Type:</b> Comprehensive Clinical Assessment",
            self.styles['Info']
        )
        elements.append(metadata)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_patient_section(self, patient_data):
        """Create patient information section."""
        elements = []
        
        elements.append(Paragraph("Patient Information", self.styles['SectionHeader']))
        
        # Patient info table
        patient_info = [
            ['Patient ID:', patient_data.get('patient_id', 'N/A')],
            ['Name:', patient_data.get('name', 'Anonymous')],
            ['Age:', str(patient_data.get('age', 'N/A'))],
            ['Gender:', patient_data.get('gender', 'N/A')],
            ['Assessment Date:', patient_data.get('assessment_date', datetime.now().strftime('%Y-%m-%d'))],
            ['Referring Physician:', patient_data.get('referring_physician', 'N/A')]
        ]
        
        table = Table(patient_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_executive_summary(self, analysis_results):
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Calculate overall risk
        risk_score = analysis_results.get('overall_risk_score', 0.0)
        risk_level = self._get_risk_level(risk_score)
        
        # Risk assessment box
        risk_color = self._get_risk_color(risk_level)
        
        summary_text = f"""
        <b>Overall Assessment:</b> {risk_level} Risk for Parkinson's Disease<br/>
        <b>Confidence Score:</b> {risk_score * 100:.1f}%<br/>
        <b>Assessment Modalities:</b> {', '.join(analysis_results.get('modalities_tested', ['Voice', 'Handwriting', 'Facial', 'Motion']))}
        """
        
        elements.append(Paragraph(summary_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key findings
        elements.append(Paragraph("Key Findings:", self.styles['SubSection']))
        findings = analysis_results.get('key_findings', [
            'Multi-modal analysis completed successfully',
            'Results correlate across different assessment methods',
            'Clinical correlation recommended'
        ])
        
        for finding in findings:
            elements.append(Paragraph(f"• {finding}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_voice_analysis_section(self, voice_data):
        """Create voice analysis section."""
        elements = []
        
        elements.append(Paragraph("Voice Analysis Results", self.styles['SectionHeader']))
        
        # Voice parameters table
        voice_params = [
            ['Parameter', 'Value', 'Normal Range', 'Status'],
            ['Jitter (RAP)', f"{voice_data.get('jitter_rap', 0):.5f}", '0.001-0.005', 
             self._get_status(voice_data.get('jitter_rap', 0), 0.005)],
            ['Shimmer', f"{voice_data.get('shimmer', 0):.5f}", '0.01-0.04',
             self._get_status(voice_data.get('shimmer', 0), 0.04)],
            ['HNR (dB)', f"{voice_data.get('hnr', 25):.2f}", '20-30',
             'Normal' if voice_data.get('hnr', 25) > 20 else 'Abnormal'],
            ['Fundamental Frequency', f"{voice_data.get('fundamental_freq', 150):.1f} Hz", '85-180 Hz', 'Normal']
        ]
        
        table = Table(voice_params, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Interpretation
        detection_prob = voice_data.get('detection_probability', 0.0)
        interpretation = f"Voice analysis indicates {detection_prob*100:.1f}% probability of Parkinson's-related vocal dysfunction. "
        
        if detection_prob > 0.7:
            interpretation += "Significant abnormalities detected in voice parameters."
        elif detection_prob > 0.4:
            interpretation += "Moderate indicators of vocal changes present."
        else:
            interpretation += "Voice parameters within normal range."
        
        elements.append(Paragraph(interpretation, self.styles['BodyText']))
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_handwriting_section(self, handwriting_data):
        """Create handwriting analysis section."""
        elements = []
        
        elements.append(Paragraph("Handwriting Analysis Results", self.styles['SectionHeader']))
        
        # Handwriting parameters
        hw_params = [
            ['Parameter', 'Value', 'Assessment'],
            ['Tremor Frequency', f"{handwriting_data.get('tremor_frequency', 4.5):.1f} Hz", 
             'Normal' if handwriting_data.get('tremor_frequency', 4.5) < 5 else 'Elevated'],
            ['Line Smoothness', f"{handwriting_data.get('line_smoothness', 0.8):.3f}",
             'Good' if handwriting_data.get('line_smoothness', 0.8) > 0.7 else 'Impaired'],
            ['Micrographia Score', f"{handwriting_data.get('micrographia_score', 0.1):.3f}",
             'Absent' if handwriting_data.get('micrographia_score', 0.1) < 0.3 else 'Present']
        ]
        
        table = Table(hw_params, colWidths=[2.5*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_facial_section(self, facial_data):
        """Create facial analysis section."""
        elements = []
        
        elements.append(Paragraph("Facial Expression Analysis", self.styles['SectionHeader']))
        
        facial_metrics = [
            ['Metric', 'Score', 'Interpretation'],
            ['Expression Mobility', f"{facial_data.get('expression_mobility', 0.6):.3f}",
             'Normal' if facial_data.get('expression_mobility', 0.6) > 0.6 else 'Reduced'],
            ['Facial Symmetry', f"{facial_data.get('facial_symmetry', 0.85):.3f}",
             'Symmetric' if facial_data.get('facial_symmetry', 0.85) > 0.8 else 'Asymmetric'],
            ['Blink Frequency', f"{facial_data.get('blink_frequency', 18):.0f}/min",
             'Normal' if facial_data.get('blink_frequency', 18) > 15 else 'Reduced']
        ]
        
        table = Table(facial_metrics, colWidths=[2.5*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7'))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_motion_section(self, motion_data):
        """Create motion analysis section."""
        elements = []
        
        elements.append(Paragraph("Motion & Motor Function Analysis", self.styles['SectionHeader']))
        
        # Motor symptoms summary
        motor_summary = motion_data.get('motor_summary', {})
        
        symptoms_text = f"""
        <b>Motor Score:</b> {motor_summary.get('total_score', 0)}/{motor_summary.get('max_score', 20)}<br/>
        <b>Risk Level:</b> {motor_summary.get('risk_level', 'Not Assessed')}<br/>
        <b>Symptoms Present:</b> {len(motor_summary.get('symptoms_present', []))}
        """
        
        elements.append(Paragraph(symptoms_text, self.styles['BodyText']))
        elements.append(Spacer(1, 0.2*inch))
        
        # List symptoms
        if motor_summary.get('symptoms_present'):
            elements.append(Paragraph("Reported Motor Symptoms:", self.styles['SubSection']))
            for symptom in motor_summary.get('symptoms_present', []):
                symptom_name = symptom.replace('_', ' ').title()
                elements.append(Paragraph(f"• {symptom_name}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_recommendations_section(self, analysis_results):
        """Create clinical recommendations section."""
        elements = []
        
        elements.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))
        
        risk_score = analysis_results.get('overall_risk_score', 0.0)
        
        if risk_score > 0.7:
            recommendations = [
                "Urgent referral to movement disorder specialist recommended within 2-4 weeks",
                "Comprehensive neurological evaluation including UPDRS assessment",
                "Consider DaTscan imaging for dopamine transporter assessment",
                "Document baseline symptoms for monitoring disease progression",
                "Family education and support resources coordination"
            ]
        elif risk_score > 0.4:
            recommendations = [
                "Neurological consultation recommended within 3-6 months",
                "Repeat multi-modal assessment in 6 months",
                "Consider detailed clinical movement disorder evaluation",
                "Monitor symptom progression through regular assessments",
                "Lifestyle modifications: regular exercise, stress management"
            ]
        else:
            recommendations = [
                "Continue routine health maintenance and monitoring",
                "No immediate specialist referral required",
                "Repeat assessment in 12 months or if symptoms develop",
                "Maintain healthy lifestyle with regular physical activity",
                "Report any new motor symptoms to primary care physician"
            ]
        
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_disclaimer(self):
        """Create medical disclaimer section."""
        elements = []
        
        elements.append(Paragraph("Important Medical Disclaimer", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This automated assessment is based on computational analysis of motor and vocal biomarkers 
        and should be interpreted only in conjunction with comprehensive clinical examination, 
        detailed patient history, and other appropriate diagnostic modalities. The results presented 
        in this report are not definitive diagnoses and do not replace professional clinical judgment 
        by qualified healthcare providers.
        <br/><br/>
        Machine learning predictions have inherent limitations and may be affected by recording quality, 
        patient compliance, medications, comorbid conditions, and other confounding factors. This tool 
        is intended for clinical decision support only and should not be used as the sole basis for 
        diagnosis or treatment decisions.
        <br/><br/>
        For questions regarding this report or to discuss clinical implications, please consult with 
        a qualified neurologist or movement disorder specialist.
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['Info']))
        
        return elements
    
    def _get_risk_level(self, risk_score):
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return 'Very High'
        elif risk_score >= 0.6:
            return 'High'
        elif risk_score >= 0.3:
            return 'Moderate'
        else:
            return 'Low'
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level."""
        colors_map = {
            'Very High': colors.red,
            'High': colors.orange,
            'Moderate': colors.yellow,
            'Low': colors.green
        }
        return colors_map.get(risk_level, colors.gray)
    
    def _get_status(self, value, threshold):
        """Get status based on threshold."""
        return 'Normal' if value <= threshold else 'Elevated'
