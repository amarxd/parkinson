"""
Handwriting feature definitions and analysis parameters for Parkinson's Disease detection.
Based on research literature on motor dysfunction patterns in handwriting and drawing tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class DrawingTaskType(Enum):
    """Types of drawing tasks used in PD assessment."""
    SPIRAL = "spiral"
    WAVE = "wave"
    CIRCLE = "circle"
    FREE_DRAWING = "free_drawing"
    TEXT_WRITING = "text_writing"

@dataclass
class HandwritingFeatureDefinition:
    """Definition of a handwriting/drawing feature for PD detection."""
    name: str
    description: str
    normal_range: Tuple[float, float]
    units: str
    calculation_method: str
    clinical_significance: str
    pd_trend: str  # 'higher', 'lower', or 'variable'
    applicable_tasks: List[DrawingTaskType]

class HandwritingFeatureRegistry:
    """Registry of handwriting and drawing features used in Parkinson's Disease detection."""
    
    def __init__(self):
        self.features = self._initialize_features()
        self.feature_groups = self._define_feature_groups()
        self.clinical_thresholds = self._define_clinical_thresholds()
        self.task_specific_weights = self._define_task_weights()
    
    def _initialize_features(self) -> Dict[str, HandwritingFeatureDefinition]:
        """Initialize all handwriting feature definitions."""
        features = {}
        
        # Tremor-related Features
        features['tremor_frequency'] = HandwritingFeatureDefinition(
            name='Tremor Frequency',
            description='Dominant frequency of tremor oscillations in handwriting',
            normal_range=(3.0, 5.0),
            units='Hz',
            calculation_method='Spectral analysis of writing trajectory to identify dominant tremor frequency',
            clinical_significance='Elevated frequency (4-6 Hz) characteristic of PD rest tremor manifesting during writing',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['tremor_amplitude'] = HandwritingFeatureDefinition(
            name='Tremor Amplitude',
            description='Magnitude of tremor oscillations relative to intended movement',
            normal_range=(0.0, 0.2),
            units='normalized amplitude',
            calculation_method='RMS amplitude of high-frequency components in movement trajectory',
            clinical_significance='Increased amplitude indicates more severe motor dysfunction and tremor intrusion',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['tremor_consistency'] = HandwritingFeatureDefinition(
            name='Tremor Consistency',
            description='Stability of tremor frequency and amplitude over time',
            normal_range=(0.7, 1.0),
            units='consistency score',
            calculation_method='Coefficient of variation of tremor parameters across writing segments',
            clinical_significance='Reduced consistency indicates variable motor control typical of PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.TEXT_WRITING]
        )
        
        # Movement Quality Features
        features['line_smoothness'] = HandwritingFeatureDefinition(
            name='Line Smoothness',
            description='Measure of trajectory smoothness and fluidity',
            normal_range=(0.7, 1.0),
            units='smoothness index',
            calculation_method='Inverse of normalized jerk (third derivative of position)',
            clinical_significance='Reduced smoothness reflects bradykinesia and loss of fine motor control',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['velocity_consistency'] = HandwritingFeatureDefinition(
            name='Velocity Consistency',
            description='Stability of writing/drawing velocity',
            normal_range=(0.6, 0.9),
            units='consistency score',
            calculation_method='Inverse coefficient of variation of instantaneous velocity',
            clinical_significance='Irregular velocity patterns indicate motor planning difficulties in PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['acceleration_variance'] = HandwritingFeatureDefinition(
            name='Acceleration Variance',
            description='Variability in writing acceleration patterns',
            normal_range=(0.1, 0.4),
            units='normalized variance',
            calculation_method='Normalized variance of acceleration magnitude',
            clinical_significance='Increased variance reflects compromised motor planning and execution',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        # Pressure and Force Features
        features['pressure_mean'] = HandwritingFeatureDefinition(
            name='Mean Writing Pressure',
            description='Average pressure applied during writing',
            normal_range=(0.3, 0.8),
            units='normalized pressure',
            calculation_method='Mean of pressure values across all writing points',
            clinical_significance='Altered pressure may indicate rigidity or compensatory mechanisms',
            pd_trend='variable',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['pressure_variation'] = HandwritingFeatureDefinition(
            name='Pressure Variation',
            description='Variability in applied writing pressure',
            normal_range=(0.1, 0.3),
            units='coefficient of variation',
            calculation_method='Coefficient of variation of pressure measurements',
            clinical_significance='Increased variation indicates difficulty maintaining consistent force control',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['pressure_rhythm'] = HandwritingFeatureDefinition(
            name='Pressure Rhythm',
            description='Rhythmicity of pressure application patterns',
            normal_range=(0.6, 0.9),
            units='rhythm score',
            calculation_method='Autocorrelation-based measure of pressure pattern regularity',
            clinical_significance='Disrupted rhythm reflects motor sequencing difficulties',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.TEXT_WRITING, DrawingTaskType.WAVE]
        )
        
        # Spatial Features
        features['stroke_width'] = HandwritingFeatureDefinition(
            name='Average Stroke Width',
            description='Mean width of drawn strokes',
            normal_range=(2.0, 5.0),
            units='pixels',
            calculation_method='Mean width calculated from distance transform of binary image',
            clinical_significance='Progressive reduction (micrographia) is characteristic of PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['stroke_width_variability'] = HandwritingFeatureDefinition(
            name='Stroke Width Variability',
            description='Consistency of stroke width throughout drawing',
            normal_range=(0.1, 0.3),
            units='coefficient of variation',
            calculation_method='Coefficient of variation of stroke widths',
            clinical_significance='Increased variability indicates inconsistent motor output',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['spatial_coverage'] = HandwritingFeatureDefinition(
            name='Spatial Coverage',
            description='Proportion of available drawing space utilized',
            normal_range=(0.4, 0.8),
            units='coverage ratio',
            calculation_method='Ratio of drawing area to total canvas area',
            clinical_significance='Reduced coverage may indicate bradykinesia or hesitation',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.CIRCLE, DrawingTaskType.FREE_DRAWING]
        )
        
        features['drawing_density'] = HandwritingFeatureDefinition(
            name='Drawing Density',
            description='Density of ink/strokes within the drawing area',
            normal_range=(0.2, 0.6),
            units='density ratio',
            calculation_method='Ratio of inked pixels to drawing bounding box area',
            clinical_significance='Altered density may reflect motor planning changes',
            pd_trend='variable',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.CIRCLE, DrawingTaskType.FREE_DRAWING]
        )
        
        # Size and Scaling Features  
        features['size_consistency'] = HandwritingFeatureDefinition(
            name='Size Consistency',
            description='Consistency of feature sizes throughout the drawing',
            normal_range=(0.7, 0.95),
            units='consistency score',
            calculation_method='Inverse coefficient of variation of local feature sizes',
            clinical_significance='Progressive size reduction (micrographia) characteristic of PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.TEXT_WRITING]
        )
        
        features['micrographia_progression'] = HandwritingFeatureDefinition(
            name='Micrographia Progression',
            description='Progressive reduction in writing/drawing size',
            normal_range=(0.0, 0.3),
            units='progression score',
            calculation_method='Slope of linear regression of feature sizes over time/position',
            clinical_significance='Progressive size reduction is pathognomonic for PD',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.TEXT_WRITING]
        )
        
        features['aspect_ratio_stability'] = HandwritingFeatureDefinition(
            name='Aspect Ratio Stability',
            description='Stability of height-to-width ratios in drawn features',
            normal_range=(0.7, 0.95),
            units='stability score',
            calculation_method='Inverse coefficient of variation of local aspect ratios',
            clinical_significance='Instability indicates compromised spatial motor control',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        # Timing and Speed Features
        features['writing_speed'] = HandwritingFeatureDefinition(
            name='Average Writing Speed',
            description='Mean velocity of pen/stylus movement',
            normal_range=(80.0, 150.0),
            units='pixels/second',
            calculation_method='Mean instantaneous velocity across all movement segments',
            clinical_significance='Reduced speed (bradykinesia) is core feature of PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['speed_variability'] = HandwritingFeatureDefinition(
            name='Speed Variability',
            description='Variability in writing/drawing speed',
            normal_range=(0.2, 0.5),
            units='coefficient of variation',
            calculation_method='Coefficient of variation of instantaneous velocities',
            clinical_significance='Increased variability reflects motor planning difficulties',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['pause_frequency'] = HandwritingFeatureDefinition(
            name='Pause Frequency',
            description='Frequency of movement pauses during writing',
            normal_range=(0.0, 0.3),
            units='pauses per second',
            calculation_method='Number of velocity drops below threshold per unit time',
            clinical_significance='Increased pauses indicate motor freezing or hesitation',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['pause_duration'] = HandwritingFeatureDefinition(
            name='Average Pause Duration',
            description='Mean duration of movement pauses',
            normal_range=(0.0, 0.5),
            units='seconds',
            calculation_method='Mean duration of periods with velocity below threshold',
            clinical_significance='Prolonged pauses suggest motor planning difficulties',
            pd_trend='higher',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        # Task-Specific Features
        features['spiral_regularity'] = HandwritingFeatureDefinition(
            name='Spiral Regularity',
            description='Regularity of spiral geometry and spacing',
            normal_range=(0.6, 0.9),
            units='regularity score',
            calculation_method='Measure of deviation from ideal Archimedean spiral',
            clinical_significance='Irregular spirals indicate compromised visuomotor control',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL]
        )
        
        features['spiral_tightness'] = HandwritingFeatureDefinition(
            name='Spiral Tightness',
            description='Consistency of spacing between spiral turns',
            normal_range=(0.7, 0.95),
            units='tightness score',
            calculation_method='Coefficient of variation of inter-turn distances',
            clinical_significance='Irregular tightness reflects motor control deterioration',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL]
        )
        
        features['completion_rate'] = HandwritingFeatureDefinition(
            name='Task Completion Rate',
            description='Proportion of intended task successfully completed',
            normal_range=(0.85, 1.0),
            units='completion ratio',
            calculation_method='Ratio of completed to intended task elements',
            clinical_significance='Reduced completion indicates fatigue or motor dysfunction',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE]
        )
        
        features['symmetry_score'] = HandwritingFeatureDefinition(
            name='Drawing Symmetry',
            description='Bilateral symmetry of drawn shapes',
            normal_range=(0.6, 0.9),
            units='symmetry score',
            calculation_method='Cross-correlation of shape halves after reflection',
            clinical_significance='Reduced symmetry indicates unilateral motor dysfunction',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.CIRCLE, DrawingTaskType.SPIRAL]
        )
        
        features['wave_amplitude_consistency'] = HandwritingFeatureDefinition(
            name='Wave Amplitude Consistency',
            description='Consistency of wave peak amplitudes',
            normal_range=(0.7, 0.95),
            units='consistency score',
            calculation_method='Inverse coefficient of variation of wave peak amplitudes',
            clinical_significance='Inconsistent amplitudes reflect motor planning difficulties',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.WAVE]
        )
        
        features['wave_frequency_stability'] = HandwritingFeatureDefinition(
            name='Wave Frequency Stability',
            description='Stability of wave frequency throughout drawing',
            normal_range=(0.7, 0.95),
            units='stability score',
            calculation_method='Inverse coefficient of variation of local wave frequencies',
            clinical_significance='Frequency instability indicates motor rhythm disruption',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.WAVE]
        )
        
        # Advanced Motor Control Features
        features['motor_planning_score'] = HandwritingFeatureDefinition(
            name='Motor Planning Score',
            description='Overall measure of motor planning effectiveness',
            normal_range=(0.6, 0.9),
            units='planning score',
            calculation_method='Composite score from velocity profiles and path efficiency',
            clinical_significance='Reduced planning reflects basal ganglia dysfunction in PD',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['motor_execution_score'] = HandwritingFeatureDefinition(
            name='Motor Execution Score',
            description='Quality of motor execution relative to apparent intent',
            normal_range=(0.6, 0.9),
            units='execution score',
            calculation_method='Composite score from smoothness, accuracy, and consistency measures',
            clinical_significance='Poor execution indicates compromised motor output pathways',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.WAVE, DrawingTaskType.CIRCLE, DrawingTaskType.TEXT_WRITING]
        )
        
        features['visuomotor_coordination'] = HandwritingFeatureDefinition(
            name='Visuomotor Coordination',
            description='Integration of visual guidance with motor output',
            normal_range=(0.6, 0.9),
            units='coordination score',
            calculation_method='Measure of drawing accuracy relative to visual target',
            clinical_significance='Poor coordination reflects impaired sensorimotor integration',
            pd_trend='lower',
            applicable_tasks=[DrawingTaskType.SPIRAL, DrawingTaskType.CIRCLE, DrawingTaskType.FREE_DRAWING]
        )
        
        return features
    
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define logical groupings of handwriting features."""
        return {
            'tremor_features': [
                'tremor_frequency', 'tremor_amplitude', 'tremor_consistency'
            ],
            'movement_quality': [
                'line_smoothness', 'velocity_consistency', 'acceleration_variance'
            ],
            'pressure_dynamics': [
                'pressure_mean', 'pressure_variation', 'pressure_rhythm'
            ],
            'spatial_features': [
                'stroke_width', 'stroke_width_variability', 'spatial_coverage', 'drawing_density'
            ],
            'size_scaling': [
                'size_consistency', 'micrographia_progression', 'aspect_ratio_stability'
            ],
            'timing_features': [
                'writing_speed', 'speed_variability', 'pause_frequency', 'pause_duration'
            ],
            'spiral_specific': [
                'spiral_regularity', 'spiral_tightness'
            ],
            'wave_specific': [
                'wave_amplitude_consistency', 'wave_frequency_stability'
            ],
            'task_completion': [
                'completion_rate', 'symmetry_score'
            ],
            'motor_control': [
                'motor_planning_score', 'motor_execution_score', 'visuomotor_coordination'
            ]
        }
    
    def _define_clinical_thresholds(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Define clinical interpretation thresholds for handwriting features."""
        return {
            'mild_pd': {
                'tremor_frequency': (5.0, 6.0),
                'tremor_amplitude': (0.2, 0.35),
                'line_smoothness': (0.5, 0.7),
                'velocity_consistency': (0.4, 0.6),
                'pressure_variation': (0.3, 0.45),
                'writing_speed': (60.0, 80.0),
                'size_consistency': (0.5, 0.7),
                'micrographia_progression': (0.3, 0.5),
                'completion_rate': (0.7, 0.85),
                'symmetry_score': (0.4, 0.6)
            },
            'moderate_pd': {
                'tremor_frequency': (6.0, 7.0),
                'tremor_amplitude': (0.35, 0.55),
                'line_smoothness': (0.3, 0.5),
                'velocity_consistency': (0.2, 0.4),
                'pressure_variation': (0.45, 0.65),
                'writing_speed': (40.0, 60.0),
                'size_consistency': (0.3, 0.5),
                'micrographia_progression': (0.5, 0.7),
                'completion_rate': (0.5, 0.7),
                'symmetry_score': (0.2, 0.4)
            },
            'severe_pd': {
                'tremor_frequency': (7.0, float('inf')),
                'tremor_amplitude': (0.55, 1.0),
                'line_smoothness': (0.0, 0.3),
                'velocity_consistency': (0.0, 0.2),
                'pressure_variation': (0.65, 1.0),
                'writing_speed': (0.0, 40.0),
                'size_consistency': (0.0, 0.3),
                'micrographia_progression': (0.7, 1.0),
                'completion_rate': (0.0, 0.5),
                'symmetry_score': (0.0, 0.2)
            }
        }
    
    def _define_task_weights(self) -> Dict[DrawingTaskType, Dict[str, float]]:
        """Define feature importance weights for different drawing tasks."""
        return {
            DrawingTaskType.SPIRAL: {
                'tremor_frequency': 0.95,
                'tremor_amplitude': 0.90,
                'line_smoothness': 0.88,
                'spiral_regularity': 0.92,
                'spiral_tightness': 0.85,
                'micrographia_progression': 0.90,
                'writing_speed': 0.80,
                'completion_rate': 0.85,
                'size_consistency': 0.88
            },
            DrawingTaskType.WAVE: {
                'tremor_frequency': 0.88,
                'tremor_amplitude': 0.85,
                'line_smoothness': 0.90,
                'wave_amplitude_consistency': 0.92,
                'wave_frequency_stability': 0.88,
                'velocity_consistency': 0.85,
                'writing_speed': 0.80,
                'pressure_rhythm': 0.83
            },
            DrawingTaskType.CIRCLE: {
                'tremor_frequency': 0.85,
                'line_smoothness': 0.88,
                'symmetry_score': 0.92,
                'completion_rate': 0.87,
                'aspect_ratio_stability': 0.90,
                'writing_speed': 0.78,
                'visuomotor_coordination': 0.85
            },
            DrawingTaskType.TEXT_WRITING: {
                'micrographia_progression': 0.95,
                'size_consistency': 0.92,
                'pressure_variation': 0.88,
                'writing_speed': 0.90,
                'tremor_frequency': 0.85,
                'line_smoothness': 0.83,
                'pause_frequency': 0.80
            }
        }
    
    def get_feature_definition(self, feature_name: str) -> Optional[HandwritingFeatureDefinition]:
        """Get definition for a specific feature."""
        return self.features.get(feature_name)
    
    def get_feature_group(self, group_name: str) -> List[str]:
        """Get list of features in a specific group."""
        return self.feature_groups.get(group_name, [])
    
    def get_all_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.features.keys())
    
    def get_task_specific_features(self, task_type: DrawingTaskType) -> List[str]:
        """Get features applicable to a specific drawing task."""
        applicable_features = []
        for name, feature in self.features.items():
            if task_type in feature.applicable_tasks:
                applicable_features.append(name)
        return applicable_features
    
    def get_pd_sensitive_features(self) -> List[str]:
        """Get features most sensitive to Parkinson's Disease changes."""
        pd_sensitive = []
        for name, feature in self.features.items():
            if feature.pd_trend in ['higher', 'lower']:
                pd_sensitive.append(name)
        return pd_sensitive
    
    def classify_feature_abnormality(self, feature_name: str, value: float) -> str:
        """
        Classify feature value as normal, mild, moderate, or severe abnormality.
        
        Args:
            feature_name: Name of the feature
            value: Measured feature value
            
        Returns:
            str: Classification ('normal', 'mild', 'moderate', 'severe')
        """
        if feature_name not in self.features:
            return 'unknown'
        
        feature_def = self.features[feature_name]
        normal_range = feature_def.normal_range
        
        # Check if value is in normal range
        if normal_range[0] <= value <= normal_range[1]:
            return 'normal'
        
        # For features that increase in PD (worse)
        if feature_def.pd_trend == 'higher':
            if value > normal_range[1]:
                thresholds = self.clinical_thresholds
                if value <= thresholds['mild_pd'].get(feature_name, (float('inf'), float('inf')))[1]:
                    return 'mild'
                elif value <= thresholds['moderate_pd'].get(feature_name, (float('inf'), float('inf')))[1]:
                    return 'moderate'
                else:
                    return 'severe'
        
        # For features that decrease in PD (worse)
        elif feature_def.pd_trend == 'lower':
            if value < normal_range[0]:
                thresholds = self.clinical_thresholds
                if value >= thresholds['mild_pd'].get(feature_name, (0, 0))[0]:
                    return 'mild'
                elif value >= thresholds['moderate_pd'].get(feature_name, (0, 0))[0]:
                    return 'moderate'
                else:
                    return 'severe'
        
        return 'normal'
    
    def get_clinical_interpretation(self, feature_name: str, value: float, 
                                  task_type: Optional[DrawingTaskType] = None) -> Dict[str, str]:
        """
        Get clinical interpretation of a feature value.
        
        Args:
            feature_name: Name of the feature
            value: Measured feature value
            task_type: Type of drawing task (optional)
            
        Returns:
            Dict containing interpretation details
        """
        if feature_name not in self.features:
            return {'interpretation': 'Unknown feature', 'significance': 'N/A'}
        
        feature_def = self.features[feature_name]
        classification = self.classify_feature_abnormality(feature_name, value)
        
        # Task-specific interpretation modifiers
        task_context = ""
        if task_type and task_type in feature_def.applicable_tasks:
            task_weights = self.task_specific_weights.get(task_type, {})
            feature_weight = task_weights.get(feature_name, 0.5)
            if feature_weight > 0.8:
                task_context = f" (highly significant for {task_type.value} tasks)"
            elif feature_weight > 0.6:
                task_context = f" (moderately significant for {task_type.value} tasks)"
        
        interpretation_map = {
            'normal': f'{feature_def.name} within normal limits{task_context}',
            'mild': f'Mildly abnormal {feature_def.name} - possible early motor dysfunction{task_context}',
            'moderate': f'Moderately abnormal {feature_def.name} - consistent with motor impairment{task_context}',
            'severe': f'Severely abnormal {feature_def.name} - significant motor dysfunction{task_context}'
        }
        
        return {
            'interpretation': interpretation_map.get(classification, 'Unknown classification'),
            'significance': feature_def.clinical_significance,
            'classification': classification,
            'normal_range': f'{feature_def.normal_range[0]} - {feature_def.normal_range[1]} {feature_def.units}',
            'measured_value': f'{value:.3f} {feature_def.units}',
            'task_relevance': task_context.strip()
        }
    
    def get_feature_importance_weights(self, task_type: Optional[DrawingTaskType] = None) -> Dict[str, float]:
        """
        Get relative importance weights for features.
        
        Args:
            task_type: Specific drawing task to get weights for
            
        Returns:
            Dict mapping feature names to importance weights (0.0-1.0)
        """
        if task_type and task_type in self.task_specific_weights:
            return self.task_specific_weights[task_type]
        
        # General weights based on discriminative power in PD literature
        return {
            # Highest importance - consistently discriminative
            'micrographia_progression': 0.95,
            'tremor_frequency': 0.92,
            'line_smoothness': 0.90,
            'writing_speed': 0.88,
            'size_consistency': 0.87,
            
            # High importance
            'tremor_amplitude': 0.85,
            'completion_rate': 0.83,
            'velocity_consistency': 0.82,
            'pressure_variation': 0.80,
            'spiral_regularity': 0.82,
            
            # Medium-high importance
            'symmetry_score': 0.75,
            'pause_frequency': 0.73,
            'acceleration_variance': 0.72,
            'motor_planning_score': 0.70,
            'speed_variability': 0.70,
            
            # Medium importance
            'spatial_coverage': 0.65,
            'stroke_width_variability': 0.63,
            'tremor_consistency': 0.62,
            'motor_execution_score': 0.60,
            'wave_amplitude_consistency': 0.65,
            
            # Lower importance but still relevant
            'drawing_density': 0.55,
            'pressure_mean': 0.53,
            'aspect_ratio_stability': 0.52,
            'visuomotor_coordination': 0.50,
            'pause_duration': 0.48,
            'pressure_rhythm': 0.45,
            'stroke_width': 0.43,
            'spiral_tightness': 0.45,
            'wave_frequency_stability': 0.47
        }
    
    def calculate_composite_scores(self, feature_values: Dict[str, float], 
                                 task_type: Optional[DrawingTaskType] = None) -> Dict[str, float]:
        """
        Calculate composite scores for different aspects of motor function.
        
        Args:
            feature_values: Dictionary of feature names to values
            task_type: Type of drawing task
            
        Returns:
            Dict containing composite scores
        """
        composite_scores = {}
        
        # Tremor composite score
        tremor_features = self.get_feature_group('tremor_features')
        tremor_values = [feature_values.get(f, 0.5) for f in tremor_features if f in feature_values]
        if tremor_values:
            composite_scores['tremor_severity'] = np.mean(tremor_values)
        
        # Motor quality composite score
        motor_features = self.get_feature_group('movement_quality')
        motor_values = [feature_values.get(f, 0.7) for f in motor_features if f in feature_values]
        if motor_values:
            composite_scores['motor_quality'] = np.mean(motor_values)
        
        # Bradykinesia composite score (speed and size features)
        brady_features = ['writing_speed', 'size_consistency', 'micrographia_progression']
        brady_values = []
        for f in brady_features:
            if f in feature_values:
                value = feature_values[f]
                # Invert micrographia_progression since higher is worse
                if f == 'micrographia_progression':
                    value = 1.0 - value
                brady_values.append(value)
        
        if brady_values:
            composite_scores['bradykinesia_severity'] = 1.0 - np.mean(brady_values)
        
        # Task-specific composite scores
        if task_type:
            task_features = self.get_task_specific_features(task_type)
            task_weights = self.get_feature_importance_weights(task_type)
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for feature in task_features:
                if feature in feature_values:
                    value = feature_values[feature]
                    weight = task_weights.get(feature, 0.5)
                    
                    # Normalize value based on PD trend
                    feature_def = self.features.get(feature)
                    if feature_def and feature_def.pd_trend == 'higher':
                        # For features where higher is worse, invert
                        normal_max = feature_def.normal_range[1]
                        normalized_value = max(0, min(1, 1.0 - (value - normal_max) / normal_max))
                    else:
                        # For features where lower is worse
                        normal_min = feature_def.normal_range[0] if feature_def else 0.5
                        normalized_value = max(0, min(1, value / normal_min)) if normal_min > 0 else 0.5
                    
                    weighted_sum += normalized_value * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                composite_scores[f'{task_type.value}_performance'] = weighted_sum / weight_sum
        
        return composite_scores
    
    def validate_feature_values(self, feature_values: Dict[str, float], 
                              task_type: Optional[DrawingTaskType] = None) -> Dict[str, List[str]]:
        """
        Validate feature values against expected ranges and return any issues.
        
        Args:
            feature_values: Dictionary of feature names to values
            task_type: Type of drawing task (optional)
            
        Returns:
            Dict containing validation results and any warnings/errors
        """
        validation_results = {
            'valid_features': [],
            'warnings': [],
            'errors': [],
            'task_compatibility': []
        }
        
        for feature_name, value in feature_values.items():
            if feature_name not in self.features:
                validation_results['warnings'].append(f'Unknown feature: {feature_name}')
                continue
            
            feature_def = self.features[feature_name]
            
            # Check for valid numeric value
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                validation_results['errors'].append(f'Invalid value for {feature_name}: {value}')
                continue
            
            # Check task compatibility
            if task_type and task_type not in feature_def.applicable_tasks:
                validation_results['task_compatibility'].append(
                    f'{feature_name} may not be relevant for {task_type.value} tasks'
                )
            
            # Check for reasonable physiological ranges
            normal_min, normal_max = feature_def.normal_range
            extended_min = max(0, normal_min - 5 * abs(normal_min))
            extended_max = normal_max + 5 * abs(normal_max)
            
            if not (extended_min <= value <= extended_max):
                validation_results['warnings'].append(
                    f'{feature_name} value ({value:.3f}) outside expected range '
                    f'({extended_min:.3f} - {extended_max:.3f})'
                )
            
            validation_results['valid_features'].append(feature_name)
        
        return validation_results

# Global instance for easy access
HANDWRITING_FEATURES = HandwritingFeatureRegistry()

# Convenience functions
def get_handwriting_feature_definition(feature_name: str) -> Optional[HandwritingFeatureDefinition]:
    """Get definition for a handwriting feature."""
    return HANDWRITING_FEATURES.get_feature_definition(feature_name)

def get_task_features(task_type: DrawingTaskType) -> List[str]:
    """Get features applicable to a specific drawing task."""
    return HANDWRITING_FEATURES.get_task_specific_features(task_type)

def get_pd_sensitive_handwriting_features() -> List[str]:
    """Get list of handwriting features most sensitive to PD changes."""
    return HANDWRITING_FEATURES.get_pd_sensitive_features()

def classify_handwriting_abnormality(feature_name: str, value: float) -> str:
    """Classify a handwriting feature value abnormality level."""
    return HANDWRITING_FEATURES.classify_feature_abnormality(feature_name, value)

def get_handwriting_interpretation(feature_name: str, value: float, 
                                 task_type: Optional[DrawingTaskType] = None) -> Dict[str, str]:
    """Get clinical interpretation of a handwriting feature value."""
    return HANDWRITING_FEATURES.get_clinical_interpretation(feature_name, value, task_type)

def get_handwriting_feature_importance(task_type: Optional[DrawingTaskType] = None) -> Dict[str, float]:
    """Get feature importance weights for handwriting analysis."""
    return HANDWRITING_FEATURES.get_feature_importance_weights(task_type)

def calculate_handwriting_composite_scores(feature_values: Dict[str, float], 
                                         task_type: Optional[DrawingTaskType] = None) -> Dict[str, float]:
    """Calculate composite scores for handwriting motor function."""
    return HANDWRITING_FEATURES.calculate_composite_scores(feature_values, task_type)

def validate_handwriting_features(feature_values: Dict[str, float], 
                                task_type: Optional[DrawingTaskType] = None) -> Dict[str, List[str]]:
    """Validate handwriting feature values."""
    return HANDWRITING_FEATURES.validate_feature_values(feature_values, task_type)

# Analysis parameter constants
HANDWRITING_ANALYSIS_PARAMS = {
    'min_stroke_length': 10,      # Minimum pixels for valid stroke
    'max_pause_duration': 2.0,    # Maximum pause duration in seconds
    'pressure_threshold': 0.1,    # Minimum pressure for pen-down detection
    'smoothing_window': 5,        # Window size for velocity smoothing
    'tremor_freq_range': (3, 8),  # Frequency range for tremor detection (Hz)
    'sampling_rate': 100,         # Expected sampling rate (Hz)
    'min_drawing_points': 50,     # Minimum points for valid analysis
    'spiral_min_turns': 2,        # Minimum turns for spiral analysis
    'wave_min_cycles': 3,         # Minimum cycles for wave analysis
}

# Clinical reference ranges for quick lookup
HANDWRITING_CLINICAL_RANGES = {
    'tremor_frequency_normal': (3.0, 5.0),
    'tremor_frequency_mild_pd': (5.0, 6.0),
    'tremor_frequency_severe_pd': (6.0, 8.0),
    
    'writing_speed_normal': (80.0, 150.0),
    'writing_speed_mild_pd': (60.0, 80.0),
    'writing_speed_severe_pd': (0.0, 60.0),
    
    'line_smoothness_normal': (0.7, 1.0),
    'line_smoothness_mild_pd': (0.5, 0.7),
    'line_smoothness_severe_pd': (0.0, 0.5),
    
    'micrographia_normal': (0.0, 0.3),
    'micrographia_mild_pd': (0.3, 0.5),
    'micrographia_severe_pd': (0.5, 1.0),
}

# Task-specific analysis guidelines
TASK_ANALYSIS_GUIDELINES = {
    DrawingTaskType.SPIRAL: {
        'min_turns': 2,
        'max_turns': 6,
        'preferred_direction': 'outward',
        'key_features': ['spiral_regularity', 'tremor_frequency', 'micrographia_progression'],
        'analysis_focus': 'Progressive motor control and scaling'
    },
    
    DrawingTaskType.WAVE: {
        'min_cycles': 3,
        'max_cycles': 10,
        'preferred_orientation': 'horizontal',
        'key_features': ['wave_amplitude_consistency', 'tremor_frequency', 'line_smoothness'],
        'analysis_focus': 'Rhythmic motor control and coordination'
    },
    
    DrawingTaskType.CIRCLE: {
        'diameter_range': (50, 200),
        'closure_tolerance': 0.1,
        'key_features': ['symmetry_score', 'line_smoothness', 'completion_rate'],
        'analysis_focus': 'Continuous motor control and spatial accuracy'
    },
    
    DrawingTaskType.TEXT_WRITING: {
        'min_characters': 5,
        'line_spacing': 'auto',
        'key_features': ['micrographia_progression', 'writing_speed', 'pressure_variation'],
        'analysis_focus': 'Fine motor control and progressive deterioration'
    }
}
