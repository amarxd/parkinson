"""
Voice feature definitions and extraction parameters for Parkinson's Disease detection.
Based on research literature and UCI Oxford Parkinson Dataset specifications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class VoiceFeatureDefinition:
    """Definition of a voice feature for PD detection."""
    name: str
    description: str
    normal_range: Tuple[float, float]
    units: str
    calculation_method: str
    clinical_significance: str
    pd_trend: str  # 'higher', 'lower', or 'variable'

class VoiceFeatureRegistry:
    """Registry of voice features used in Parkinson's Disease detection."""
    
    def __init__(self):
        self.features = self._initialize_features()
        self.feature_groups = self._define_feature_groups()
        self.clinical_thresholds = self._define_clinical_thresholds()
    
    def _initialize_features(self) -> Dict[str, VoiceFeatureDefinition]:
        """Initialize all voice feature definitions."""
        features = {}
        
        # Jitter (Frequency Perturbation) Features
        features['jitter_local'] = VoiceFeatureDefinition(
            name='Local Jitter',
            description='Cycle-to-cycle variation in fundamental frequency',
            normal_range=(0.001, 0.004),
            units='%',
            calculation_method='Average absolute difference between consecutive periods, normalized by mean period',
            clinical_significance='Reflects vocal fold tension irregularities and neuromuscular control',
            pd_trend='higher'
        )
        
        features['jitter_rap'] = VoiceFeatureDefinition(
            name='Jitter RAP (Relative Average Perturbation)',
            description='Three-point period perturbation quotient',
            normal_range=(0.001, 0.005),
            units='%',
            calculation_method='Average of relative differences between period and mean of it and its neighbors',
            clinical_significance='Sensitive indicator of vocal fold vibration irregularities in PD',
            pd_trend='higher'
        )
        
        features['jitter_ppq5'] = VoiceFeatureDefinition(
            name='Jitter PPQ5 (5-point Period Perturbation Quotient)',
            description='Five-point period perturbation quotient',
            normal_range=(0.001, 0.005),
            units='%',
            calculation_method='Average of relative differences between period and mean of 5-period neighborhood',
            clinical_significance='Reflects longer-term frequency stability affected by motor control',
            pd_trend='higher'
        )
        
        features['jitter_ddp'] = VoiceFeatureDefinition(
            name='Jitter DDP (Difference of Differences of Periods)',
            description='Variation in consecutive period differences',
            normal_range=(0.003, 0.015),
            units='%',
            calculation_method='Average absolute difference of differences between consecutive periods',
            clinical_significance='Captures acceleration patterns in frequency perturbation',
            pd_trend='higher'
        )
        
        # Shimmer (Amplitude Perturbation) Features
        features['shimmer_local'] = VoiceFeatureDefinition(
            name='Local Shimmer',
            description='Cycle-to-cycle variation in amplitude',
            normal_range=(0.01, 0.04),
            units='%',
            calculation_method='Average absolute difference between consecutive amplitudes, normalized by mean',
            clinical_significance='Reflects laryngeal muscle control and vocal fold tension variations',
            pd_trend='higher'
        )
        
        features['shimmer_apq3'] = VoiceFeatureDefinition(
            name='Shimmer APQ3 (3-point Amplitude Perturbation Quotient)',
            description='Three-point amplitude perturbation quotient',
            normal_range=(0.01, 0.03),
            units='%',
            calculation_method='Average of relative differences between amplitude and mean of 3-point neighborhood',
            clinical_significance='Sensitive to short-term amplitude control deficits in PD',
            pd_trend='higher'
        )
        
        features['shimmer_apq5'] = VoiceFeatureDefinition(
            name='Shimmer APQ5 (5-point Amplitude Perturbation Quotient)',
            description='Five-point amplitude perturbation quotient',
            normal_range=(0.01, 0.035),
            units='%',
            calculation_method='Average of relative differences between amplitude and mean of 5-point neighborhood',
            clinical_significance='Reflects medium-term amplitude stability in voice production',
            pd_trend='higher'
        )
        
        features['shimmer_apq11'] = VoiceFeatureDefinition(
            name='Shimmer APQ11 (11-point Amplitude Perturbation Quotient)',
            description='Eleven-point amplitude perturbation quotient',
            normal_range=(0.015, 0.045),
            units='%',
            calculation_method='Average of relative differences between amplitude and mean of 11-point neighborhood',
            clinical_significance='Captures longer-term amplitude control patterns',
            pd_trend='higher'
        )
        
        features['shimmer_dda'] = VoiceFeatureDefinition(
            name='Shimmer DDA (Difference of Differences of Amplitudes)',
            description='Variation in consecutive amplitude differences',
            normal_range=(0.03, 0.09),
            units='%',
            calculation_method='Average absolute difference of differences between consecutive amplitudes',
            clinical_significance='Reflects amplitude acceleration patterns in voice control',
            pd_trend='higher'
        )
        
        # Noise Measures
        features['nhr'] = VoiceFeatureDefinition(
            name='NHR (Noise-to-Harmonics Ratio)',
            description='Ratio of inharmonic to harmonic energy in voice signal',
            normal_range=(0.01, 0.15),
            units='ratio',
            calculation_method='Ratio of energy in noise frequency band to harmonic energy',
            clinical_significance='Increased noise indicates vocal fold pathology and reduced motor control',
            pd_trend='higher'
        )
        
        features['hnr'] = VoiceFeatureDefinition(
            name='HNR (Harmonics-to-Noise Ratio)',
            description='Ratio of harmonic to inharmonic energy in voice signal',
            normal_range=(20.0, 30.0),
            units='dB',
            calculation_method='Logarithmic ratio of harmonic energy to noise energy',
            clinical_significance='Reduced HNR indicates voice quality degradation in PD',
            pd_trend='lower'
        )
        
        # Fundamental Frequency Features
        features['f0_mean'] = VoiceFeatureDefinition(
            name='Mean Fundamental Frequency',
            description='Average fundamental frequency across voice sample',
            normal_range=(100.0, 150.0),
            units='Hz',
            calculation_method='Mean of fundamental frequency estimates across analysis frames',
            clinical_significance='May be reduced in PD due to vocal fold stiffness',
            pd_trend='lower'
        )
        
        features['f0_std'] = VoiceFeatureDefinition(
            name='F0 Standard Deviation',
            description='Variability in fundamental frequency',
            normal_range=(10.0, 25.0),
            units='Hz',
            calculation_method='Standard deviation of fundamental frequency estimates',
            clinical_significance='Increased variability reflects reduced pitch control in PD',
            pd_trend='higher'
        )
        
        features['f0_min'] = VoiceFeatureDefinition(
            name='Minimum Fundamental Frequency',
            description='Lowest fundamental frequency in voice sample',
            normal_range=(80.0, 120.0),
            units='Hz',
            calculation_method='Minimum detected fundamental frequency',
            clinical_significance='May indicate vocal fold tension limitations',
            pd_trend='lower'
        )
        
        features['f0_max'] = VoiceFeatureDefinition(
            name='Maximum Fundamental Frequency',
            description='Highest fundamental frequency in voice sample',
            normal_range=(120.0, 180.0),
            units='Hz',
            calculation_method='Maximum detected fundamental frequency',
            clinical_significance='Reduced range may indicate motor limitations',
            pd_trend='lower'
        )
        
        # Nonlinear Dynamical Analysis Features
        features['rpde'] = VoiceFeatureDefinition(
            name='RPDE (Recurrence Period Density Entropy)',
            description='Measure of signal complexity and predictability',
            normal_range=(0.2, 0.5),
            units='entropy units',
            calculation_method='Entropy of recurrence period distribution in phase space',
            clinical_significance='Increased RPDE reflects reduced vocal control complexity in PD',
            pd_trend='higher'
        )
        
        features['dfa'] = VoiceFeatureDefinition(
            name='DFA (Detrended Fluctuation Analysis)',
            description='Measure of long-term correlations in voice signal',
            normal_range=(0.6, 0.9),
            units='scaling exponent',
            calculation_method='Scaling exponent of detrended fluctuation analysis',
            clinical_significance='Altered DFA indicates disrupted vocal motor patterns in PD',
            pd_trend='lower'
        )
        
        features['ppe'] = VoiceFeatureDefinition(
            name='PPE (Pitch Period Entropy)',
            description='Entropy of pitch period variation',
            normal_range=(0.1, 0.3),
            units='entropy units',
            calculation_method='Shannon entropy of pitch period distribution',
            clinical_significance='Increased PPE reflects irregular pitch control in PD',
            pd_trend='higher'
        )
        
        # Spectral Features
        features['spectral_centroid'] = VoiceFeatureDefinition(
            name='Spectral Centroid',
            description='Center of mass of spectral energy distribution',
            normal_range=(1000.0, 2000.0),
            units='Hz',
            calculation_method='Weighted mean of spectral frequencies',
            clinical_significance='May be altered due to vocal tract changes in PD',
            pd_trend='lower'
        )
        
        features['spectral_rolloff'] = VoiceFeatureDefinition(
            name='Spectral Rolloff',
            description='Frequency below which 85% of spectral energy is contained',
            normal_range=(2000.0, 4000.0),
            units='Hz',
            calculation_method='Frequency threshold for 85% cumulative spectral energy',
            clinical_significance='Reflects articulatory precision affected in PD',
            pd_trend='lower'
        )
        
        features['spectral_bandwidth'] = VoiceFeatureDefinition(
            name='Spectral Bandwidth',
            description='Width of spectral energy distribution',
            normal_range=(1000.0, 2500.0),
            units='Hz',
            calculation_method='Weighted standard deviation of spectral frequencies',
            clinical_significance='May indicate changes in voice quality and articulation',
            pd_trend='variable'
        )
        
        features['zero_crossing_rate'] = VoiceFeatureDefinition(
            name='Zero Crossing Rate',
            description='Rate at which signal crosses zero amplitude',
            normal_range=(0.05, 0.15),
            units='crossings/frame',
            calculation_method='Number of zero crossings per unit time',
            clinical_significance='Reflects voice periodicity and noise characteristics',
            pd_trend='higher'
        )
        
        # MFCC Features (summary statistics)
        features['mfcc_mean'] = VoiceFeatureDefinition(
            name='MFCC Mean',
            description='Mean of Mel-Frequency Cepstral Coefficients',
            normal_range=(-15.0, 0.0),
            units='coefficient value',
            calculation_method='Mean across MFCC coefficient vectors',
            clinical_significance='Reflects overall spectral shape changes in PD voice',
            pd_trend='lower'
        )
        
        features['mfcc_std'] = VoiceFeatureDefinition(
            name='MFCC Standard Deviation',
            description='Variability of Mel-Frequency Cepstral Coefficients',
            normal_range=(8.0, 20.0),
            units='coefficient value',
            calculation_method='Standard deviation across MFCC coefficient vectors',
            clinical_significance='Indicates spectral variability changes in PD',
            pd_trend='higher'
        )
        
        return features
    
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """Define logical groupings of voice features."""
        return {
            'jitter_features': [
                'jitter_local', 'jitter_rap', 'jitter_ppq5', 'jitter_ddp'
            ],
            'shimmer_features': [
                'shimmer_local', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda'
            ],
            'noise_features': [
                'nhr', 'hnr'
            ],
            'fundamental_frequency': [
                'f0_mean', 'f0_std', 'f0_min', 'f0_max'
            ],
            'nonlinear_features': [
                'rpde', 'dfa', 'ppe'
            ],
            'spectral_features': [
                'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate'
            ],
            'cepstral_features': [
                'mfcc_mean', 'mfcc_std'
            ]
        }
    
    def _define_clinical_thresholds(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Define clinical interpretation thresholds for voice features."""
        return {
            'mild_pd': {
                'jitter_rap': (0.005, 0.008),
                'shimmer_local': (0.04, 0.07),
                'nhr': (0.15, 0.25),
                'hnr': (15.0, 20.0),
                'rpde': (0.5, 0.65),
                'dfa': (0.5, 0.6),
                'ppe': (0.3, 0.45)
            },
            'moderate_pd': {
                'jitter_rap': (0.008, 0.012),
                'shimmer_local': (0.07, 0.12),
                'nhr': (0.25, 0.4),
                'hnr': (10.0, 15.0),
                'rpde': (0.65, 0.8),
                'dfa': (0.4, 0.5),
                'ppe': (0.45, 0.6)
            },
            'severe_pd': {
                'jitter_rap': (0.012, float('inf')),
                'shimmer_local': (0.12, float('inf')),
                'nhr': (0.4, float('inf')),
                'hnr': (0.0, 10.0),
                'rpde': (0.8, 1.0),
                'dfa': (0.0, 0.4),
                'ppe': (0.6, 1.0)
            }
        }
    
    def get_feature_definition(self, feature_name: str) -> Optional[VoiceFeatureDefinition]:
        """Get definition for a specific feature."""
        return self.features.get(feature_name)
    
    def get_feature_group(self, group_name: str) -> List[str]:
        """Get list of features in a specific group."""
        return self.feature_groups.get(group_name, [])
    
    def get_all_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return list(self.features.keys())
    
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
        
        # For features that increase in PD
        if feature_def.pd_trend == 'higher':
            if value > normal_range[1]:
                thresholds = self.clinical_thresholds
                if value <= thresholds['mild_pd'].get(feature_name, (float('inf'), float('inf')))[1]:
                    return 'mild'
                elif value <= thresholds['moderate_pd'].get(feature_name, (float('inf'), float('inf')))[1]:
                    return 'moderate'
                else:
                    return 'severe'
            else:
                return 'below_normal'
        
        # For features that decrease in PD
        elif feature_def.pd_trend == 'lower':
            if value < normal_range[0]:
                thresholds = self.clinical_thresholds
                if value >= thresholds['mild_pd'].get(feature_name, (0, 0))[0]:
                    return 'mild'
                elif value >= thresholds['moderate_pd'].get(feature_name, (0, 0))[0]:
                    return 'moderate'
                else:
                    return 'severe'
            else:
                return 'above_normal'
        
        return 'variable'
    
    def get_clinical_interpretation(self, feature_name: str, value: float) -> Dict[str, str]:
        """
        Get clinical interpretation of a feature value.
        
        Args:
            feature_name: Name of the feature
            value: Measured feature value
            
        Returns:
            Dict containing interpretation details
        """
        if feature_name not in self.features:
            return {'interpretation': 'Unknown feature', 'significance': 'N/A'}
        
        feature_def = self.features[feature_name]
        classification = self.classify_feature_abnormality(feature_name, value)
        
        interpretation_map = {
            'normal': f'{feature_def.name} within normal limits',
            'mild': f'Mildly abnormal {feature_def.name} - possible early motor dysfunction',
            'moderate': f'Moderately abnormal {feature_def.name} - consistent with motor impairment',
            'severe': f'Severely abnormal {feature_def.name} - significant motor dysfunction',
            'below_normal': f'{feature_def.name} below normal range',
            'above_normal': f'{feature_def.name} above normal range',
            'variable': f'{feature_def.name} shows variability'
        }
        
        return {
            'interpretation': interpretation_map.get(classification, 'Unknown classification'),
            'significance': feature_def.clinical_significance,
            'classification': classification,
            'normal_range': f'{feature_def.normal_range[0]} - {feature_def.normal_range[1]} {feature_def.units}',
            'measured_value': f'{value:.6f} {feature_def.units}' if feature_def.units != 'ratio' else f'{value:.6f}'
        }
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Get relative importance weights for features based on research literature.
        
        Returns:
            Dict mapping feature names to importance weights (0.0-1.0)
        """
        # Weights based on discriminative power reported in PD literature
        return {
            # High importance - consistently reported as most discriminative
            'jitter_rap': 0.95,
            'shimmer_local': 0.92,
            'nhr': 0.88,
            'hnr': 0.87,
            'rpde': 0.85,
            
            # Medium-high importance
            'jitter_ppq5': 0.80,
            'shimmer_apq3': 0.78,
            'dfa': 0.75,
            'ppe': 0.73,
            'shimmer_apq5': 0.70,
            
            # Medium importance
            'jitter_ddp': 0.65,
            'shimmer_apq11': 0.63,
            'shimmer_dda': 0.60,
            'f0_std': 0.58,
            'spectral_centroid': 0.55,
            
            # Lower importance
            'f0_mean': 0.50,
            'spectral_rolloff': 0.48,
            'spectral_bandwidth': 0.45,
            'zero_crossing_rate': 0.43,
            'f0_min': 0.40,
            'f0_max': 0.40,
            'mfcc_mean': 0.38,
            'mfcc_std': 0.35
        }
    
    def validate_feature_values(self, feature_values: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Validate feature values against expected ranges and return any issues.
        
        Args:
            feature_values: Dictionary of feature names to values
            
        Returns:
            Dict containing validation results and any warnings/errors
        """
        validation_results = {
            'valid_features': [],
            'warnings': [],
            'errors': []
        }
        
        for feature_name, value in feature_values.items():
            if feature_name not in self.features:
                validation_results['warnings'].append(f'Unknown feature: {feature_name}')
                continue
            
            # Check for valid numeric value
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                validation_results['errors'].append(f'Invalid value for {feature_name}: {value}')
                continue
            
            # Check for reasonable physiological ranges (extended beyond normal)
            feature_def = self.features[feature_name]
            normal_min, normal_max = feature_def.normal_range
            
            # Define extended physiological limits (10x normal range)
            extended_min = max(0, normal_min - 10 * abs(normal_min))
            extended_max = normal_max + 10 * abs(normal_max)
            
            if not (extended_min <= value <= extended_max):
                validation_results['warnings'].append(
                    f'{feature_name} value ({value:.6f}) outside expected physiological range '
                    f'({extended_min:.6f} - {extended_max:.6f})'
                )
            
            validation_results['valid_features'].append(feature_name)
        
        return validation_results

# Global instance for easy access
VOICE_FEATURES = VoiceFeatureRegistry()

# Convenience functions
def get_feature_definition(feature_name: str) -> Optional[VoiceFeatureDefinition]:
    """Get definition for a voice feature."""
    return VOICE_FEATURES.get_feature_definition(feature_name)

def get_pd_sensitive_features() -> List[str]:
    """Get list of features most sensitive to PD changes."""
    return VOICE_FEATURES.get_pd_sensitive_features()

def classify_voice_abnormality(feature_name: str, value: float) -> str:
    """Classify a voice feature value abnormality level."""
    return VOICE_FEATURES.classify_feature_abnormality(feature_name, value)

def get_voice_interpretation(feature_name: str, value: float) -> Dict[str, str]:
    """Get clinical interpretation of a voice feature value."""
    return VOICE_FEATURES.get_clinical_interpretation(feature_name, value)

def get_feature_importance() -> Dict[str, float]:
    """Get feature importance weights for voice analysis."""
    return VOICE_FEATURES.get_feature_importance_weights()

def validate_voice_features(feature_values: Dict[str, float]) -> Dict[str, List[str]]:
    """Validate voice feature values."""
    return VOICE_FEATURES.validate_feature_values(feature_values)

# Feature extraction parameter constants
VOICE_ANALYSIS_PARAMS = {
    'sample_rate': 22050,
    'frame_length': 2048,
    'hop_length': 512,
    'window_size': 0.04,  # 40ms windows for F0 estimation
    'hop_size': 0.01,     # 10ms hop for F0 estimation
    'f0_min': 50,         # Minimum F0 for detection
    'f0_max': 400,        # Maximum F0 for detection
    'n_mfcc': 13,         # Number of MFCC coefficients
    'pre_emphasis': 0.97, # Pre-emphasis filter coefficient
}

# Clinical reference ranges for quick lookup
CLINICAL_REFERENCE_RANGES = {
    'jitter_rap_normal': (0.001, 0.005),
    'jitter_rap_mild_pd': (0.005, 0.008),
    'jitter_rap_severe_pd': (0.008, float('inf')),
    
    'shimmer_normal': (0.01, 0.04),
    'shimmer_mild_pd': (0.04, 0.07),
    'shimmer_severe_pd': (0.07, float('inf')),
    
    'nhr_normal': (0.01, 0.15),
    'nhr_mild_pd': (0.15, 0.25),
    'nhr_severe_pd': (0.25, float('inf')),
    
    'hnr_normal': (20.0, 30.0),
    'hnr_mild_pd': (15.0, 20.0),
    'hnr_severe_pd': (0.0, 15.0),
}
