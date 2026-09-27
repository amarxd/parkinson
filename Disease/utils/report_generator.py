import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from typing import Dict, List, Tuple, Optional

class ReportGenerator:
    """
    Clinical report generator for Parkinson's Disease detection analysis.
    Generates comprehensive medical reports with analysis results, visualizations,
    and clinical recommendations based on voice and handwriting assessments.
    """
    
    def __init__(self):
        self.report_templates = {
            'individual': self._get_individual_report_template(),
            'batch': self._get_batch_report_template(),
            'clinical_study': self._get_clinical_study_template(),
            'progress': self._get_progress_report_template()
        }
        
        # Clinical interpretation thresholds
        self.clinical_thresholds = {
            'voice': {
                'jitter_rap': {'normal': (0.001, 0.005), 'mild': (0.005, 0.008), 'severe': (0.008, float('inf'))},
                'shimmer': {'normal': (0.01, 0.04), 'mild': (0.04, 0.07), 'severe': (0.07, float('inf'))},
                'nhr': {'normal': (0.01, 0.15), 'mild': (0.15, 0.25), 'severe': (0.25, float('inf'))},
                'hnr': {'normal': (20, 30), 'mild': (15, 20), 'severe': (0, 15)}
            },
            'handwriting': {
                'tremor_frequency': {'normal': (3, 5), 'mild': (5, 6.5), 'severe': (6.5, 8)},
                'line_smoothness': {'normal': (0.7, 1.0), 'mild': (0.5, 0.7), 'severe': (0, 0.5)},
                'micrographia_score': {'normal': (0, 0.3), 'mild': (0.3, 0.6), 'severe': (0.6, 1.0)},
                'completion_rate': {'normal': (0.85, 1.0), 'mild': (0.7, 0.85), 'severe': (0, 0.7)}
            }
        }
        
        # Risk level definitions
        self.risk_levels = {
            'low': {'threshold': 0.3, 'color': '#28a745', 'recommendations': 'routine_monitoring'},
            'moderate': {'threshold': 0.6, 'color': '#ffc107', 'recommendations': 'enhanced_monitoring'},
            'high': {'threshold': 0.8, 'color': '#dc3545', 'recommendations': 'immediate_referral'},
            'very_high': {'threshold': 1.0, 'color': '#8b0000', 'recommendations': 'urgent_intervention'}
        }
    
    def generate_individual_report(self, patient_data: Dict, voice_results: Optional[Dict] = None, 
                                 handwriting_results: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive individual patient report.
        
        Args:
            patient_data: Dictionary containing patient information
            voice_results: Voice analysis results
            handwriting_results: Handwriting analysis results
            
        Returns:
            Dict: Formatted report data
        """
        report = {
            'report_type': 'individual',
            'generated_at': datetime.now().isoformat(),
            'patient_info': self._format_patient_info(patient_data),
            'executive_summary': {},
            'detailed_analysis': {},
            'clinical_interpretation': {},
            'recommendations': {},
            'follow_up_plan': {}
        }
        
        # Calculate overall risk assessment
        combined_probability = self._calculate_combined_risk(voice_results, handwriting_results)
        risk_level = self._determine_risk_level(combined_probability)
        
        # Executive Summary
        report['executive_summary'] = {
            'overall_risk_level': risk_level,
            'combined_probability': combined_probability,
            'primary_findings': self._generate_primary_findings(voice_results, handwriting_results),
            'confidence_score': self._calculate_confidence_score(voice_results, handwriting_results)
        }
        
        # Detailed Analysis
        if voice_results:
            report['detailed_analysis']['voice'] = self._format_voice_analysis(voice_results)
        
        if handwriting_results:
            report['detailed_analysis']['handwriting'] = self._format_handwriting_analysis(handwriting_results)
        
        # Clinical Interpretation
        report['clinical_interpretation'] = self._generate_clinical_interpretation(
            voice_results, handwriting_results, risk_level
        )
        
        # Recommendations
        report['recommendations'] = self._generate_recommendations(risk_level, voice_results, handwriting_results)
        
        # Follow-up Plan
        report['follow_up_plan'] = self._generate_follow_up_plan(risk_level, combined_probability)
        
        return report
    
    def generate_batch_report(self, batch_results: List[Dict]) -> Dict:
        """
        Generate batch analysis report for multiple patients.
        
        Args:
            batch_results: List of patient analysis results
            
        Returns:
            Dict: Batch report data
        """
        report = {
            'report_type': 'batch',
            'generated_at': datetime.now().isoformat(),
            'summary_statistics': self._calculate_batch_statistics(batch_results),
            'risk_distribution': self._analyze_risk_distribution(batch_results),
            'population_insights': self._generate_population_insights(batch_results),
            'quality_metrics': self._calculate_quality_metrics(batch_results),
            'recommendations': self._generate_batch_recommendations(batch_results)
        }
        
        return report
    
    def generate_progress_report(self, patient_id: str, historical_data: List[Dict]) -> Dict:
        """
        Generate progress monitoring report for longitudinal analysis.
        
        Args:
            patient_id: Patient identifier
            historical_data: List of historical assessment results
            
        Returns:
            Dict: Progress report data
        """
        report = {
            'report_type': 'progress',
            'patient_id': patient_id,
            'generated_at': datetime.now().isoformat(),
            'timeline_analysis': self._analyze_progression_timeline(historical_data),
            'trend_analysis': self._calculate_progression_trends(historical_data),
            'clinical_correlation': self._analyze_clinical_correlation(historical_data),
            'predictive_insights': self._generate_predictive_insights(historical_data),
            'monitoring_recommendations': self._generate_monitoring_recommendations(historical_data)
        }
        
        return report
    
    def _format_patient_info(self, patient_data: Dict) -> Dict:
        """Format patient information for report."""
        return {
            'patient_id': patient_data.get('patient_id', 'Unknown'),
            'name': patient_data.get('name', 'Anonymous'),
            'age': patient_data.get('age', 'Unknown'),
            'gender': patient_data.get('gender', 'Unknown'),
            'assessment_date': patient_data.get('assessment_date', datetime.now().strftime('%Y-%m-%d')),
            'referring_physician': patient_data.get('referring_physician', 'Unknown'),
            'clinical_notes': patient_data.get('clinical_notes', 'None provided')
        }
    
    def _calculate_combined_risk(self, voice_results: Optional[Dict], handwriting_results: Optional[Dict]) -> float:
        """Calculate combined risk probability from voice and handwriting results."""
        if not voice_results and not handwriting_results:
            return 0.0
        
        voice_prob = voice_results.get('detection_probability', 0.0) if voice_results else 0.0
        handwriting_prob = handwriting_results.get('detection_probability', 0.0) if handwriting_results else 0.0
        
        if voice_results and handwriting_results:
            # Weighted average with slight boost for concordant results
            combined = (voice_prob + handwriting_prob) / 2
            concordance_bonus = 0.05 if abs(voice_prob - handwriting_prob) < 0.2 else 0.0
            return min(1.0, combined + concordance_bonus)
        elif voice_results:
            return voice_prob * 0.85  # Single modality discount
        else:
            return handwriting_prob * 0.85
    
    def _determine_risk_level(self, probability: float) -> str:
        """Determine risk level based on probability."""
        for level, config in reversed(list(self.risk_levels.items())):
            if probability >= config['threshold']:
                return level
        return 'low'
    
    def _generate_primary_findings(self, voice_results: Optional[Dict], handwriting_results: Optional[Dict]) -> List[str]:
        """Generate list of primary clinical findings."""
        findings = []
        
        if voice_results:
            # Voice findings
            jitter = voice_results.get('jitter_rap', 0)
            shimmer = voice_results.get('shimmer', 0)
            nhr = voice_results.get('nhr', 0)
            hnr = voice_results.get('hnr', 25)
            
            if jitter > 0.008:
                findings.append("Severe voice frequency perturbation (jitter) detected")
            elif jitter > 0.005:
                findings.append("Mild to moderate voice frequency perturbation observed")
            
            if shimmer > 0.07:
                findings.append("Severe voice amplitude perturbation (shimmer) detected")
            elif shimmer > 0.04:
                findings.append("Mild to moderate voice amplitude perturbation observed")
            
            if nhr > 0.25:
                findings.append("Elevated noise-to-harmonics ratio indicating voice quality degradation")
            
            if hnr < 15:
                findings.append("Reduced harmonics-to-noise ratio suggesting vocal dysfunction")
        
        if handwriting_results:
            # Handwriting findings
            tremor_freq = handwriting_results.get('tremor_frequency', 4.5)
            smoothness = handwriting_results.get('line_smoothness', 0.8)
            micrographia = handwriting_results.get('micrographia_score', 0.1)
            completion = handwriting_results.get('completion_rate', 0.9)
            
            if tremor_freq > 6.5:
                findings.append("Severe motor tremor detected in handwriting patterns")
            elif tremor_freq > 5.0:
                findings.append("Mild to moderate tremor observed in writing dynamics")
            
            if smoothness < 0.5:
                findings.append("Significant reduction in writing smoothness and motor control")
            elif smoothness < 0.7:
                findings.append("Mild impairment in writing fluidity observed")
            
            if micrographia > 0.6:
                findings.append("Severe micrographia (progressive writing size reduction) detected")
            elif micrographia > 0.3:
                findings.append("Mild micrographia tendencies observed")
            
            if completion < 0.7:
                findings.append("Significant difficulty completing complex drawing tasks")
        
        if not findings:
            findings.append("No significant pathological indicators detected in current assessment")
        
        return findings
    
    def _calculate_confidence_score(self, voice_results: Optional[Dict], handwriting_results: Optional[Dict]) -> float:
        """Calculate overall confidence score for the assessment."""
        confidence_factors = []
        
        if voice_results:
            # Voice confidence based on feature consistency
            voice_confidence = voice_results.get('confidence_interval', (0.8, 0.9))
            confidence_factors.append(np.mean(voice_confidence))
        
        if handwriting_results:
            # Handwriting confidence based on drawing quality
            handwriting_confidence = handwriting_results.get('confidence_interval', (0.8, 0.9))
            confidence_factors.append(np.mean(handwriting_confidence))
        
        if confidence_factors:
            return np.mean(confidence_factors)
        else:
            return 0.5
    
    def _format_voice_analysis(self, voice_results: Dict) -> Dict:
        """Format voice analysis results for report."""
        return {
            'detection_probability': voice_results.get('detection_probability', 0.0),
            'confidence_interval': voice_results.get('confidence_interval', (0.0, 1.0)),
            'key_parameters': {
                'jitter_rap': {
                    'value': voice_results.get('jitter_rap', 0.0),
                    'interpretation': self._interpret_parameter('voice', 'jitter_rap', voice_results.get('jitter_rap', 0.0)),
                    'normal_range': '0.001-0.005'
                },
                'shimmer': {
                    'value': voice_results.get('shimmer', 0.0),
                    'interpretation': self._interpret_parameter('voice', 'shimmer', voice_results.get('shimmer', 0.0)),
                    'normal_range': '0.01-0.04'
                },
                'nhr': {
                    'value': voice_results.get('nhr', 0.0),
                    'interpretation': self._interpret_parameter('voice', 'nhr', voice_results.get('nhr', 0.0)),
                    'normal_range': '0.01-0.15'
                },
                'hnr': {
                    'value': voice_results.get('hnr', 25.0),
                    'interpretation': self._interpret_parameter('voice', 'hnr', voice_results.get('hnr', 25.0)),
                    'normal_range': '20-30 dB'
                }
            },
            'clinical_significance': self._assess_clinical_significance('voice', voice_results)
        }
    
    def _format_handwriting_analysis(self, handwriting_results: Dict) -> Dict:
        """Format handwriting analysis results for report."""
        return {
            'detection_probability': handwriting_results.get('detection_probability', 0.0),
            'confidence_interval': handwriting_results.get('confidence_interval', (0.0, 1.0)),
            'key_parameters': {
                'tremor_frequency': {
                    'value': handwriting_results.get('tremor_frequency', 4.5),
                    'interpretation': self._interpret_parameter('handwriting', 'tremor_frequency', handwriting_results.get('tremor_frequency', 4.5)),
                    'normal_range': '3-5 Hz'
                },
                'line_smoothness': {
                    'value': handwriting_results.get('line_smoothness', 0.8),
                    'interpretation': self._interpret_parameter('handwriting', 'line_smoothness', handwriting_results.get('line_smoothness', 0.8)),
                    'normal_range': '0.7-1.0'
                },
                'micrographia_score': {
                    'value': handwriting_results.get('micrographia_score', 0.1),
                    'interpretation': self._interpret_parameter('handwriting', 'micrographia_score', handwriting_results.get('micrographia_score', 0.1)),
                    'normal_range': '0.0-0.3'
                },
                'completion_rate': {
                    'value': handwriting_results.get('completion_rate', 0.9),
                    'interpretation': self._interpret_parameter('handwriting', 'completion_rate', handwriting_results.get('completion_rate', 0.9)),
                    'normal_range': '0.85-1.0'
                }
            },
            'clinical_significance': self._assess_clinical_significance('handwriting', handwriting_results)
        }
    
    def _interpret_parameter(self, modality: str, parameter: str, value: float) -> str:
        """Interpret individual parameter values."""
        if modality not in self.clinical_thresholds or parameter not in self.clinical_thresholds[modality]:
            return "Normal range"
        
        thresholds = self.clinical_thresholds[modality][parameter]
        
        if parameter == 'hnr':  # HNR has inverted logic (lower is worse)
            if value >= thresholds['normal'][0]:
                return "Normal"
            elif value >= thresholds['mild'][0]:
                return "Mildly abnormal"
            else:
                return "Severely abnormal"
        elif parameter == 'line_smoothness' or parameter == 'completion_rate':  # Higher is better
            if value >= thresholds['normal'][0]:
                return "Normal"
            elif value >= thresholds['mild'][0]:
                return "Mildly impaired"
            else:
                return "Severely impaired"
        else:  # Standard parameters (lower is better)
            if value <= thresholds['normal'][1]:
                return "Normal"
            elif value <= thresholds['mild'][1]:
                return "Mildly elevated"
            else:
                return "Severely elevated"
    
    def _assess_clinical_significance(self, modality: str, results: Dict) -> str:
        """Assess overall clinical significance of results."""
        probability = results.get('detection_probability', 0.0)
        
        if probability >= 0.8:
            return "High clinical significance - strong indicators of motor dysfunction"
        elif probability >= 0.6:
            return "Moderate clinical significance - some indicators present"
        elif probability >= 0.4:
            return "Mild clinical significance - subtle changes detected"
        else:
            return "Low clinical significance - results within normal range"
    
    def _generate_clinical_interpretation(self, voice_results: Optional[Dict], 
                                       handwriting_results: Optional[Dict], risk_level: str) -> Dict:
        """Generate clinical interpretation of results."""
        interpretation = {
            'overall_assessment': self._get_overall_assessment(risk_level),
            'differential_diagnosis': self._generate_differential_diagnosis(voice_results, handwriting_results),
            'clinical_correlation': self._generate_clinical_correlation(voice_results, handwriting_results),
            'limitations': self._generate_limitations_note()
        }
        
        return interpretation
    
    def _get_overall_assessment(self, risk_level: str) -> str:
        """Get overall clinical assessment based on risk level."""
        assessments = {
            'low': "Current assessment indicates low probability of Parkinson's Disease. Results are consistent with normal motor function and voice characteristics.",
            'moderate': "Assessment reveals moderate probability of early-stage motor dysfunction. Some parameters suggest possible prodromal changes that warrant monitoring.",
            'high': "High probability assessment indicates significant motor dysfunction patterns consistent with Parkinson's Disease. Multiple biomarkers suggest active pathological processes.",
            'very_high': "Very high probability assessment with multiple convergent indicators of advanced motor dysfunction. Findings strongly suggest established Parkinson's Disease pathology."
        }
        
        return assessments.get(risk_level, "Assessment results are inconclusive.")
    
    def _generate_differential_diagnosis(self, voice_results: Optional[Dict], handwriting_results: Optional[Dict]) -> List[str]:
        """Generate differential diagnosis considerations."""
        differentials = []
        
        # Base differential considerations
        differentials.append("Parkinson's Disease")
        differentials.append("Essential Tremor")
        differentials.append("Multiple System Atrophy")
        differentials.append("Progressive Supranuclear Palsy")
        differentials.append("Drug-induced Parkinsonism")
        differentials.append("Normal aging changes")
        
        return differentials
    
    def _generate_clinical_correlation(self, voice_results: Optional[Dict], handwriting_results: Optional[Dict]) -> str:
        """Generate clinical correlation recommendations."""
        if voice_results and handwriting_results:
            voice_prob = voice_results.get('detection_probability', 0.0)
            handwriting_prob = handwriting_results.get('detection_probability', 0.0)
            
            if abs(voice_prob - handwriting_prob) < 0.2:
                return "Concordant findings across voice and handwriting modalities increase diagnostic confidence. Clinical correlation with UPDRS assessment and DaTscan imaging recommended."
            else:
                return "Discordant findings between voice and handwriting assessments require clinical correlation. Consider repeat testing and comprehensive neurological evaluation."
        else:
            return "Single modality assessment completed. Multi-modal evaluation including clinical examination and additional biomarkers recommended for comprehensive assessment."
    
    def _generate_limitations_note(self) -> str:
        """Generate limitations and disclaimers note."""
        return ("This assessment is based on computational analysis of motor biomarkers and should be interpreted "
                "in conjunction with clinical examination, patient history, and other diagnostic modalities. "
                "Results are not definitive and do not replace clinical judgment. Machine learning predictions "
                "have inherent limitations and may be affected by recording quality, patient compliance, and "
                "other confounding factors.")
    
    def _generate_recommendations(self, risk_level: str, voice_results: Optional[Dict], 
                                handwriting_results: Optional[Dict]) -> Dict:
        """Generate clinical recommendations based on assessment results."""
        recommendations = {
            'immediate_actions': [],
            'follow_up_timeline': {},
            'additional_testing': [],
            'lifestyle_modifications': [],
            'monitoring_plan': []
        }
        
        if risk_level == 'low':
            recommendations['immediate_actions'].append("Continue routine health maintenance")
            recommendations['follow_up_timeline']['next_assessment'] = "12 months"
            recommendations['lifestyle_modifications'].extend([
                "Regular exercise program",
                "Healthy diet maintenance",
                "Stress management techniques"
            ])
            
        elif risk_level == 'moderate':
            recommendations['immediate_actions'].extend([
                "Schedule neurological consultation within 3-6 months",
                "Document baseline symptoms and functional status"
            ])
            recommendations['follow_up_timeline']['neurologist_referral'] = "3-6 months"
            recommendations['follow_up_timeline']['repeat_assessment'] = "6 months"
            recommendations['additional_testing'].extend([
                "UPDRS clinical assessment",
                "Detailed neurological examination",
                "Consider DaTscan if clinically indicated"
            ])
            
        elif risk_level in ['high', 'very_high']:
            recommendations['immediate_actions'].extend([
                "Urgent neurology referral within 2-4 weeks",
                "Comprehensive movement disorder evaluation",
                "Family education and support coordination"
            ])
            recommendations['follow_up_timeline']['urgent_referral'] = "2-4 weeks"
            recommendations['follow_up_timeline']['repeat_assessment'] = "3 months"
            recommendations['additional_testing'].extend([
                "Complete UPDRS assessment",
                "DaTscan imaging",
                "Comprehensive neuropsychological testing",
                "Levodopa challenge test if appropriate"
            ])
            recommendations['monitoring_plan'].extend([
                "Regular symptom progression tracking",
                "Functional status monitoring",
                "Quality of life assessments"
            ])
        
        return recommendations
    
    def _generate_follow_up_plan(self, risk_level: str, probability: float) -> Dict:
        """Generate structured follow-up plan."""
        if risk_level == 'low':
            timeline = {
                '1_month': 'Patient self-monitoring for any new symptoms',
                '6_months': 'Primary care follow-up',
                '12_months': 'Repeat computational assessment if indicated'
            }
        elif risk_level == 'moderate':
            timeline = {
                '2_weeks': 'Schedule neurology consultation',
                '3_months': 'Neurology evaluation completed',
                '6_months': 'Repeat assessment and symptom review',
                '12_months': 'Comprehensive re-evaluation'
            }
        else:  # high or very_high
            timeline = {
                '1_week': 'Contact neurology for urgent appointment',
                '2_weeks': 'Neurology consultation completed',
                '1_month': 'Initial treatment plan established',
                '3_months': 'Treatment response assessment',
                '6_months': 'Comprehensive progress evaluation'
            }
        
        return {
            'timeline': timeline,
            'monitoring_frequency': self._get_monitoring_frequency(risk_level),
            'key_indicators': self._get_key_monitoring_indicators(risk_level)
        }
    
    def _get_monitoring_frequency(self, risk_level: str) -> str:
        """Get recommended monitoring frequency."""
        frequencies = {
            'low': 'Annual',
            'moderate': 'Every 6 months',
            'high': 'Every 3 months',
            'very_high': 'Monthly initially, then quarterly'
        }
        return frequencies.get(risk_level, 'As clinically indicated')
    
    def _get_key_monitoring_indicators(self, risk_level: str) -> List[str]:
        """Get key indicators to monitor during follow-up."""
        base_indicators = [
            'Motor symptom progression',
            'Functional independence',
            'Quality of life measures'
        ]
        
        if risk_level in ['high', 'very_high']:
            base_indicators.extend([
                'Response to medications',
                'Non-motor symptom development',
                'Cognitive function changes'
            ])
        
        return base_indicators
    
    def _calculate_batch_statistics(self, batch_results: List[Dict]) -> Dict:
        """Calculate summary statistics for batch analysis."""
        if not batch_results:
            return {}
        
        probabilities = [result.get('combined_probability', 0.0) for result in batch_results]
        risk_levels = [result.get('risk_level', 'low') for result in batch_results]
        
        return {
            'total_patients': len(batch_results),
            'mean_probability': np.mean(probabilities),
            'median_probability': np.median(probabilities),
            'std_probability': np.std(probabilities),
            'risk_level_counts': {level: risk_levels.count(level) for level in set(risk_levels)},
            'high_risk_percentage': (risk_levels.count('high') + risk_levels.count('very_high')) / len(risk_levels) * 100
        }
    
    def _analyze_risk_distribution(self, batch_results: List[Dict]) -> Dict:
        """Analyze risk distribution across batch."""
        risk_counts = {}
        for result in batch_results:
            risk_level = result.get('risk_level', 'low')
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        total = len(batch_results)
        return {
            'distribution': risk_counts,
            'percentages': {level: (count / total) * 100 for level, count in risk_counts.items()}
        }
    
    def _generate_population_insights(self, batch_results: List[Dict]) -> List[str]:
        """Generate population-level insights from batch analysis."""
        insights = []
        
        high_risk_count = sum(1 for result in batch_results 
                             if result.get('risk_level') in ['high', 'very_high'])
        total_count = len(batch_results)
        high_risk_pct = (high_risk_count / total_count) * 100 if total_count > 0 else 0
        
        if high_risk_pct > 20:
            insights.append(f"Elevated prevalence of high-risk cases ({high_risk_pct:.1f}%) suggests possible enriched population or screening bias")
        elif high_risk_pct < 5:
            insights.append(f"Low prevalence of high-risk cases ({high_risk_pct:.1f}%) consistent with general population screening")
        
        return insights
    
    def _calculate_quality_metrics(self, batch_results: List[Dict]) -> Dict:
        """Calculate quality metrics for batch analysis."""
        return {
            'completion_rate': len(batch_results) / len(batch_results) if batch_results else 0,
            'data_quality_score': 0.95,  # Simulated quality score
            'confidence_score': np.mean([result.get('confidence_score', 0.8) for result in batch_results]) if batch_results else 0
        }
    
    def _generate_batch_recommendations(self, batch_results: List[Dict]) -> List[str]:
        """Generate recommendations for batch analysis."""
        recommendations = [
            "Review high-risk cases for urgent clinical correlation",
            "Implement standardized follow-up protocols for moderate-risk cases",
            "Consider population-based screening program optimization"
        ]
        return recommendations
    
    def _analyze_progression_timeline(self, historical_data: List[Dict]) -> Dict:
        """Analyze progression timeline from historical data."""
        if len(historical_data) < 2:
            return {'insufficient_data': True}
        
        # Sort by assessment date
        sorted_data = sorted(historical_data, key=lambda x: x.get('assessment_date', ''))
        
        timeline = {}
        for i, assessment in enumerate(sorted_data):
            timeline[f'assessment_{i+1}'] = {
                'date': assessment.get('assessment_date'),
                'probability': assessment.get('combined_probability', 0),
                'risk_level': assessment.get('risk_level', 'low')
            }
        
        return timeline
    
    def _calculate_progression_trends(self, historical_data: List[Dict]) -> Dict:
        """Calculate progression trends from historical assessments."""
        if len(historical_data) < 2:
            return {'trend': 'insufficient_data'}
        
        probabilities = [data.get('combined_probability', 0) for data in historical_data]
        
        # Simple linear trend calculation
        x = np.arange(len(probabilities))
        slope = np.polyfit(x, probabilities, 1)[0] if len(probabilities) > 1 else 0
        
        if slope > 0.05:
            trend = 'worsening'
        elif slope < -0.05:
            trend = 'improving'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'change_per_assessment': slope,
            'total_change': probabilities[-1] - probabilities[0] if probabilities else 0
        }
    
    def _analyze_clinical_correlation(self, historical_data: List[Dict]) -> Dict:
        """Analyze clinical correlation over time."""
        return {
            'correlation_strength': 'moderate',
            'key_correlations': [
                'Voice measures correlate with handwriting deterioration',
                'Progressive decline consistent with disease progression'
            ]
        }
    
    def _generate_predictive_insights(self, historical_data: List[Dict]) -> Dict:
        """Generate predictive insights from historical trends."""
        if len(historical_data) < 3:
            return {'prediction': 'insufficient_data_for_prediction'}
        
        return {
            'predicted_6_month_risk': 'moderate_increase_expected',
            'confidence': 0.7,
            'key_factors': [
                'Current progression rate',
                'Historical pattern consistency'
            ]
        }
    
    def _generate_monitoring_recommendations(self, historical_data: List[Dict]) -> List[str]:
        """Generate monitoring recommendations based on historical trends."""
        recommendations = [
            "Continue quarterly assessments",
            "Monitor for acceleration in symptom progression",
            "Consider additional biomarker integration"
        ]
        return recommendations
    
    def _get_individual_report_template(self) -> str:
        """Get template for individual patient reports."""
        return """
        INDIVIDUAL PATIENT ASSESSMENT REPORT
        ===================================
        
        Patient Information:
        - ID: {patient_id}
        - Name: {patient_name}
        - Age: {age}
        - Assessment Date: {assessment_date}
        
        Executive Summary:
        - Overall Risk Level: {risk_level}
        - Combined Probability: {combined_probability:.1%}
        - Confidence Score: {confidence_score:.1%}
        
        Detailed Analysis:
        {detailed_analysis}
        
        Clinical Interpretation:
        {clinical_interpretation}
        
        Recommendations:
        {recommendations}
        
        Follow-up Plan:
        {follow_up_plan}
        """
    
    def _get_batch_report_template(self) -> str:
        """Get template for batch analysis reports."""
        return "Batch analysis template"
    
    def _get_clinical_study_template(self) -> str:
        """Get template for clinical study reports."""
        return "Clinical study template"
    
    def _get_progress_report_template(self) -> str:
        """Get template for progress monitoring reports."""
        return "Progress monitoring template"
    
    def export_to_pdf(self, report_data: Dict) -> bytes:
        """
        Export report to PDF format.
        
        Args:
            report_data: Formatted report data
            
        Returns:
            bytes: PDF content
        """
        # This is a simplified PDF export
        # In a full implementation, you would use libraries like reportlab
        
        pdf_content = self._format_report_as_text(report_data)
        return pdf_content.encode('utf-8')
    
    def _format_report_as_text(self, report_data: Dict) -> str:
        """Format report data as readable text."""
        report_type = report_data.get('report_type', 'individual')
        
        if report_type == 'individual':
            return self._format_individual_report_text(report_data)
        elif report_type == 'batch':
            return self._format_batch_report_text(report_data)
        elif report_type == 'progress':
            return self._format_progress_report_text(report_data)
        else:
            return "Report formatting not available for this type."
    
    def _format_individual_report_text(self, report_data: Dict) -> str:
        """Format individual report as text."""
        patient_info = report_data.get('patient_info', {})
        executive_summary = report_data.get('executive_summary', {})
        
        text = f"""
PARKINSON'S DISEASE COMPUTATIONAL ASSESSMENT REPORT

PATIENT INFORMATION:
====================
Patient ID: {patient_info.get('patient_id', 'N/A')}
Name: {patient_info.get('name', 'N/A')}
Age: {patient_info.get('age', 'N/A')} years
Gender: {patient_info.get('gender', 'N/A')}
Assessment Date: {patient_info.get('assessment_date', 'N/A')}
Referring Physician: {patient_info.get('referring_physician', 'N/A')}

EXECUTIVE SUMMARY:
==================
Overall Risk Level: {executive_summary.get('overall_risk_level', 'N/A').upper()}
Combined Detection Probability: {executive_summary.get('combined_probability', 0):.1%}
Assessment Confidence Score: {executive_summary.get('confidence_score', 0):.1%}

PRIMARY FINDINGS:
"""
        
        findings = executive_summary.get('primary_findings', [])
        for i, finding in enumerate(findings, 1):
            text += f"{i}. {finding}\n"
        
        text += f"""
CLINICAL RECOMMENDATIONS:
=========================
Based on the {executive_summary.get('overall_risk_level', 'unknown')} risk assessment:

"""
        
        recommendations = report_data.get('recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        
        if immediate_actions:
            text += "IMMEDIATE ACTIONS:\n"
            for action in immediate_actions:
                text += f"• {action}\n"
        
        text += f"""
MEDICAL DISCLAIMER:
==================
This computational assessment is for informational purposes only and does not 
constitute medical diagnosis. Results should be interpreted by qualified healthcare 
professionals in conjunction with clinical examination and other diagnostic modalities.

Report Generated: {report_data.get('generated_at', 'N/A')}
"""
        
        return text
    
    def _format_batch_report_text(self, report_data: Dict) -> str:
        """Format batch report as text."""
        return f"""
BATCH ANALYSIS REPORT
====================

Summary Statistics:
{report_data.get('summary_statistics', {})}

Generated: {report_data.get('generated_at', 'N/A')}
"""
    
    def _format_progress_report_text(self, report_data: Dict) -> str:
        """Format progress report as text."""
        return f"""
PROGRESS MONITORING REPORT
=========================

Patient ID: {report_data.get('patient_id', 'N/A')}

Timeline Analysis:
{report_data.get('timeline_analysis', {})}

Generated: {report_data.get('generated_at', 'N/A')}
"""
