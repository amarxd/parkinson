import pandas as pd
import numpy as np
import os
import urllib.request
import io
from sklearn.datasets import make_classification

class DataLoader:
    """
    Data loader for Parkinson's Disease datasets including UCI Oxford voice data,
    handwriting datasets, and clinical progression markers.
    """
    
    def __init__(self):
        self.data_dir = "data"
        self.datasets = {}
        
        # Dataset URLs and information
        self.dataset_info = {
            'uci_voice': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
                'name': 'UCI Oxford Parkinson Voice Dataset',
                'description': '195 voice recordings from 31 individuals (23 PD, 8 controls)',
                'features': 23,
                'target': 'status'
            },
            'handwriting': {
                'name': 'Handwriting Dataset',
                'description': '62 PD patients + 15 healthy controls handwriting analysis',
                'features': 15,
                'target': 'class'
            },
            'ppmi': {
                'name': 'PPMI Clinical Dataset', 
                'description': '600 instances with clinical progression markers',
                'features': 159,
                'target': 'group'
            }
        }
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize datasets
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load all available datasets."""
        try:
            # Load UCI voice dataset
            self.datasets['voice'] = self._load_uci_voice_dataset()
            
            # Load handwriting dataset (synthetic for this implementation)
            self.datasets['handwriting'] = self._load_handwriting_dataset()
            
            # Load PPMI dataset (synthetic for this implementation)
            self.datasets['ppmi'] = self._load_ppmi_dataset()
            
        except Exception as e:
            print(f"Warning: Error loading datasets: {str(e)}")
            self._create_fallback_datasets()
    
    def _load_uci_voice_dataset(self):
        """
        Load the UCI Oxford Parkinson's Disease voice dataset.
        
        Returns:
            pd.DataFrame: Voice dataset
        """
        try:
            # Try to download from UCI repository
            url = self.dataset_info['uci_voice']['url']
            
            # Download dataset
            response = urllib.request.urlopen(url)
            data = response.read().decode('utf-8')
            
            # Parse CSV data
            lines = data.strip().split('\n')
            header = lines[0].split(',')
            
            # Create DataFrame
            rows = []
            for line in lines[1:]:
                rows.append(line.split(','))
            
            df = pd.DataFrame(rows, columns=header)
            
            # Convert numeric columns
            numeric_columns = [col for col in df.columns if col not in ['name', 'status']]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert status to numeric (1 = PD, 0 = healthy)
            df['status'] = df['status'].astype(int)
            
            print(f"Successfully loaded UCI voice dataset: {len(df)} samples")
            return df
            
        except Exception as e:
            print(f"Warning: Could not load UCI dataset from URL: {str(e)}")
            return self._create_synthetic_voice_dataset()
    
    def _create_synthetic_voice_dataset(self):
        """Create synthetic voice dataset based on UCI characteristics."""
        print("Creating synthetic voice dataset based on research parameters...")
        
        np.random.seed(42)
        
        # Create synthetic data matching UCI Parkinson's dataset structure
        n_samples = 195
        n_pd = 147  # 75% PD patients (as in original dataset)
        n_healthy = n_samples - n_pd
        
        # Feature names from UCI dataset
        feature_names = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        # Generate healthy control data
        healthy_data = self._generate_healthy_voice_samples(n_healthy, feature_names)
        healthy_status = np.zeros(n_healthy, dtype=int)
        
        # Generate PD patient data
        pd_data = self._generate_pd_voice_samples(n_pd, feature_names)
        pd_status = np.ones(n_pd, dtype=int)
        
        # Combine data
        all_data = np.vstack([healthy_data, pd_data])
        all_status = np.hstack([healthy_status, pd_status])
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=feature_names)
        df['status'] = all_status
        
        # Add patient names
        df['name'] = [f'phon_R01_S{i+1:02d}' for i in range(len(df))]
        
        # Reorder columns to match UCI format
        columns = ['name'] + feature_names + ['status']
        df = df[columns]
        
        return df
    
    def _generate_healthy_voice_samples(self, n_samples, feature_names):
        """Generate healthy voice samples with realistic parameter ranges."""
        np.random.seed(42)
        
        # Healthy parameter ranges based on literature
        healthy_params = {
            'MDVP:Fo(Hz)': (120, 25),      # Fundamental frequency
            'MDVP:Fhi(Hz)': (140, 30),     # Highest frequency
            'MDVP:Flo(Hz)': (100, 20),     # Lowest frequency
            'MDVP:Jitter(%)': (0.005, 0.002),  # Jitter percentage
            'MDVP:Jitter(Abs)': (0.00003, 0.00001),  # Absolute jitter
            'MDVP:RAP': (0.003, 0.001),    # RAP jitter
            'MDVP:PPQ': (0.003, 0.001),    # PPQ jitter
            'Jitter:DDP': (0.009, 0.003),  # DDP jitter
            'MDVP:Shimmer': (0.025, 0.01), # Shimmer
            'MDVP:Shimmer(dB)': (0.22, 0.08),  # Shimmer dB
            'Shimmer:APQ3': (0.012, 0.005),    # APQ3 shimmer
            'Shimmer:APQ5': (0.014, 0.006),    # APQ5 shimmer
            'MDVP:APQ': (0.018, 0.007),    # APQ shimmer
            'Shimmer:DDA': (0.036, 0.015), # DDA shimmer
            'NHR': (0.08, 0.03),           # Noise-to-harmonics ratio
            'HNR': (22, 3),                # Harmonics-to-noise ratio
            'RPDE': (0.45, 0.08),          # RPDE
            'DFA': (0.72, 0.06),           # DFA
            'spread1': (-6.5, 1.0),        # Spread1
            'spread2': (0.2, 0.05),        # Spread2
            'D2': (2.3, 0.3),              # D2
            'PPE': (0.18, 0.05)            # PPE
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in healthy_params:
                mean, std = healthy_params[feature]
                data[:, i] = np.random.normal(mean, std, n_samples)
            else:
                # Default values for any missing features
                data[:, i] = np.random.normal(0.5, 0.1, n_samples)
        
        # Ensure positive values where appropriate
        positive_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']
        for feature in positive_features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                data[:, idx] = np.abs(data[:, idx])
        
        return data
    
    def _generate_pd_voice_samples(self, n_samples, feature_names):
        """Generate PD patient voice samples with characteristic abnormalities."""
        np.random.seed(123)
        
        # PD parameter ranges (generally worse values)
        pd_params = {
            'MDVP:Fo(Hz)': (115, 30),      # Slightly lower, more variable
            'MDVP:Fhi(Hz)': (135, 35),     # More variable
            'MDVP:Flo(Hz)': (95, 25),      # More variable
            'MDVP:Jitter(%)': (0.008, 0.004),  # Higher jitter
            'MDVP:Jitter(Abs)': (0.00006, 0.00003),  # Higher absolute jitter
            'MDVP:RAP': (0.006, 0.003),    # Higher RAP jitter
            'MDVP:PPQ': (0.006, 0.003),    # Higher PPQ jitter
            'Jitter:DDP': (0.018, 0.008),  # Higher DDP jitter
            'MDVP:Shimmer': (0.045, 0.02), # Higher shimmer
            'MDVP:Shimmer(dB)': (0.40, 0.15),  # Higher shimmer dB
            'Shimmer:APQ3': (0.022, 0.01), # Higher APQ3 shimmer
            'Shimmer:APQ5': (0.026, 0.012), # Higher APQ5 shimmer
            'MDVP:APQ': (0.032, 0.015),    # Higher APQ shimmer
            'Shimmer:DDA': (0.066, 0.03),  # Higher DDA shimmer
            'NHR': (0.18, 0.08),           # Higher noise-to-harmonics ratio
            'HNR': (18, 4),                # Lower harmonics-to-noise ratio
            'RPDE': (0.58, 0.12),          # Higher RPDE
            'DFA': (0.65, 0.08),           # Lower DFA
            'spread1': (-5.8, 1.2),        # Different spread1
            'spread2': (0.25, 0.08),       # Different spread2
            'D2': (2.1, 0.4),              # Different D2
            'PPE': (0.25, 0.08)            # Higher PPE
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in pd_params:
                mean, std = pd_params[feature]
                data[:, i] = np.random.normal(mean, std, n_samples)
            else:
                # Default values for any missing features
                data[:, i] = np.random.normal(0.7, 0.15, n_samples)
        
        # Ensure positive values where appropriate
        positive_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'HNR']
        for feature in positive_features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                data[:, idx] = np.abs(data[:, idx])
        
        return data
    
    def _load_handwriting_dataset(self):
        """Load handwriting dataset (synthetic for this implementation)."""
        try:
            print("Creating synthetic handwriting dataset based on research parameters...")
            
            np.random.seed(456)
            
            n_samples = 77  # 62 PD + 15 controls as mentioned in research
            n_pd = 62
            n_healthy = 15
            
            # Feature names for handwriting analysis
            feature_names = [
                'tremor_frequency', 'tremor_amplitude', 'line_smoothness',
                'velocity_variation', 'stroke_width', 'size_consistency',
                'writing_speed', 'pressure_variation', 'spatial_coverage',
                'drawing_density', 'symmetry_score', 'completion_rate',
                'spiral_regularity', 'micrographia_score', 'rhythm_score'
            ]
            
            # Generate healthy samples
            healthy_data = self._generate_healthy_handwriting_samples(n_healthy, feature_names)
            healthy_class = np.zeros(n_healthy, dtype=int)
            
            # Generate PD samples
            pd_data = self._generate_pd_handwriting_samples(n_pd, feature_names)
            pd_class = np.ones(n_pd, dtype=int)
            
            # Combine data
            all_data = np.vstack([healthy_data, pd_data])
            all_class = np.hstack([healthy_class, pd_class])
            
            # Create DataFrame
            df = pd.DataFrame(all_data, columns=feature_names)
            df['class'] = all_class
            df['subject_id'] = [f'HW_{i+1:03d}' for i in range(len(df))]
            
            return df
            
        except Exception as e:
            print(f"Error creating handwriting dataset: {str(e)}")
            return self._create_fallback_handwriting_dataset()
    
    def _generate_healthy_handwriting_samples(self, n_samples, feature_names):
        """Generate healthy handwriting samples."""
        # Healthy handwriting parameters
        healthy_params = {
            'tremor_frequency': (4.2, 0.8),    # Lower tremor frequency
            'tremor_amplitude': (0.15, 0.05),  # Lower tremor amplitude
            'line_smoothness': (0.82, 0.08),   # Higher line smoothness
            'velocity_variation': (0.25, 0.05), # Lower velocity variation
            'stroke_width': (3.2, 0.5),        # Normal stroke width
            'size_consistency': (0.88, 0.06),  # Higher size consistency
            'writing_speed': (120, 15),        # Normal writing speed
            'pressure_variation': (0.20, 0.05), # Lower pressure variation
            'spatial_coverage': (0.65, 0.1),   # Good spatial coverage
            'drawing_density': (0.35, 0.08),   # Normal drawing density
            'symmetry_score': (0.85, 0.08),    # Higher symmetry
            'completion_rate': (0.92, 0.05),   # Higher completion rate
            'spiral_regularity': (0.78, 0.1),  # Higher spiral regularity
            'micrographia_score': (0.12, 0.05), # Lower micrographia score
            'rhythm_score': (0.82, 0.08)       # Higher rhythm score
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in healthy_params:
                mean, std = healthy_params[feature]
                data[:, i] = np.random.normal(mean, std, n_samples)
            else:
                data[:, i] = np.random.normal(0.5, 0.1, n_samples)
        
        # Clip values to appropriate ranges
        data[:, [2, 5, 8, 9, 10, 11, 12, 14]] = np.clip(data[:, [2, 5, 8, 9, 10, 11, 12, 14]], 0, 1)  # 0-1 scores
        data[:, 0] = np.clip(data[:, 0], 3.0, 6.0)  # tremor frequency
        data[:, 4] = np.clip(data[:, 4], 1.0, 6.0)  # stroke width
        data[:, 6] = np.clip(data[:, 6], 50, 200)   # writing speed
        
        return data
    
    def _generate_pd_handwriting_samples(self, n_samples, feature_names):
        """Generate PD handwriting samples with characteristic abnormalities."""
        # PD handwriting parameters (worse performance)
        pd_params = {
            'tremor_frequency': (5.8, 1.2),    # Higher tremor frequency
            'tremor_amplitude': (0.35, 0.1),   # Higher tremor amplitude
            'line_smoothness': (0.58, 0.12),   # Lower line smoothness
            'velocity_variation': (0.45, 0.1), # Higher velocity variation
            'stroke_width': (2.8, 0.6),        # Smaller stroke width (micrographia)
            'size_consistency': (0.65, 0.12),  # Lower size consistency
            'writing_speed': (85, 20),         # Slower writing speed
            'pressure_variation': (0.38, 0.08), # Higher pressure variation
            'spatial_coverage': (0.45, 0.12),  # Lower spatial coverage
            'drawing_density': (0.28, 0.08),   # Lower drawing density
            'symmetry_score': (0.62, 0.12),    # Lower symmetry
            'completion_rate': (0.78, 0.1),    # Lower completion rate
            'spiral_regularity': (0.52, 0.15), # Lower spiral regularity
            'micrographia_score': (0.42, 0.1), # Higher micrographia score
            'rhythm_score': (0.58, 0.12)       # Lower rhythm score
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in pd_params:
                mean, std = pd_params[feature]
                data[:, i] = np.random.normal(mean, std, n_samples)
            else:
                data[:, i] = np.random.normal(0.3, 0.15, n_samples)
        
        # Clip values to appropriate ranges
        data[:, [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]] = np.clip(data[:, [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14]], 0, 1)
        data[:, 0] = np.clip(data[:, 0], 4.0, 8.0)  # tremor frequency
        data[:, 4] = np.clip(data[:, 4], 1.5, 5.0)  # stroke width
        data[:, 6] = np.clip(data[:, 6], 40, 150)   # writing speed
        
        return data
    
    def _load_ppmi_dataset(self):
        """Load PPMI clinical dataset (synthetic for this implementation)."""
        try:
            print("Creating synthetic PPMI clinical dataset...")
            
            np.random.seed(789)
            
            n_samples = 600  # As mentioned in research
            n_pd = 400
            n_controls = 200
            
            # Clinical feature names
            feature_names = [
                'age_at_enrollment', 'disease_duration', 'updrs_total',
                'updrs_motor', 'updrs_bradykinesia', 'updrs_tremor',
                'updrs_rigidity', 'moca_score', 'semantic_fluency',
                'datscan_binding', 'csf_alpha_synuclein', 'csf_tau',
                'led_total', 'schwab_england', 'hoehn_yahr_stage'
            ]
            
            # Generate control samples
            control_data = self._generate_control_clinical_samples(n_controls, feature_names)
            control_group = ['Control'] * n_controls
            
            # Generate PD samples
            pd_data = self._generate_pd_clinical_samples(n_pd, feature_names)
            pd_group = ['PD'] * n_pd
            
            # Combine data
            all_data = np.vstack([control_data, pd_data])
            all_groups = control_group + pd_group
            
            # Create DataFrame
            df = pd.DataFrame(all_data, columns=feature_names)
            df['group'] = all_groups
            df['subject_id'] = [f'PPMI_{i+1:04d}' for i in range(len(df))]
            
            return df
            
        except Exception as e:
            print(f"Error creating PPMI dataset: {str(e)}")
            return self._create_fallback_ppmi_dataset()
    
    def _generate_control_clinical_samples(self, n_samples, feature_names):
        """Generate healthy control clinical samples."""
        control_params = {
            'age_at_enrollment': (62, 8),   # Age
            'disease_duration': (0, 0),     # No disease duration
            'updrs_total': (5, 3),          # Very low UPDRS
            'updrs_motor': (2, 2),          # Very low motor score
            'updrs_bradykinesia': (0.5, 0.5), # Minimal bradykinesia
            'updrs_tremor': (0.2, 0.3),     # Minimal tremor
            'updrs_rigidity': (0.3, 0.4),   # Minimal rigidity
            'moca_score': (28, 2),          # High cognitive score
            'semantic_fluency': (22, 4),    # Good fluency
            'datscan_binding': (2.8, 0.3),  # Normal DaTscan
            'csf_alpha_synuclein': (350, 50), # Normal levels
            'csf_tau': (45, 10),            # Normal tau
            'led_total': (0, 0),            # No medications
            'schwab_england': (95, 5),      # High functional score
            'hoehn_yahr_stage': (0, 0)      # No disease stage
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in control_params:
                mean, std = control_params[feature]
                if std == 0:
                    data[:, i] = mean
                else:
                    data[:, i] = np.random.normal(mean, std, n_samples)
        
        return data
    
    def _generate_pd_clinical_samples(self, n_samples, feature_names):
        """Generate PD patient clinical samples."""
        pd_params = {
            'age_at_enrollment': (65, 10),  # Slightly older
            'disease_duration': (5, 3),     # Years since diagnosis
            'updrs_total': (28, 12),        # Higher UPDRS total
            'updrs_motor': (18, 8),         # Higher motor score
            'updrs_bradykinesia': (6, 3),   # Significant bradykinesia
            'updrs_tremor': (4, 2),         # Tremor present
            'updrs_rigidity': (3, 2),       # Rigidity present
            'moca_score': (25, 3),          # Slightly lower cognitive
            'semantic_fluency': (18, 5),    # Reduced fluency
            'datscan_binding': (1.8, 0.4),  # Reduced DaTscan binding
            'csf_alpha_synuclein': (280, 60), # Lower alpha-synuclein
            'csf_tau': (55, 15),            # Higher tau
            'led_total': (450, 200),        # Medication dose
            'schwab_england': (78, 12),     # Reduced functional score
            'hoehn_yahr_stage': (2.2, 0.8)  # Disease stage
        }
        
        data = np.zeros((n_samples, len(feature_names)))
        
        for i, feature in enumerate(feature_names):
            if feature in pd_params:
                mean, std = pd_params[feature]
                data[:, i] = np.random.normal(mean, std, n_samples)
        
        # Ensure appropriate ranges
        data[:, feature_names.index('moca_score')] = np.clip(data[:, feature_names.index('moca_score')], 10, 30)
        data[:, feature_names.index('schwab_england')] = np.clip(data[:, feature_names.index('schwab_england')], 30, 100)
        data[:, feature_names.index('hoehn_yahr_stage')] = np.clip(data[:, feature_names.index('hoehn_yahr_stage')], 1, 5)
        
        return data
    
    def _create_fallback_datasets(self):
        """Create minimal fallback datasets."""
        self.datasets['voice'] = self._create_fallback_voice_dataset()
        self.datasets['handwriting'] = self._create_fallback_handwriting_dataset()
        self.datasets['ppmi'] = self._create_fallback_ppmi_dataset()
    
    def _create_fallback_voice_dataset(self):
        """Create minimal fallback voice dataset."""
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['status'] = y
        df['name'] = [f'sample_{i}' for i in range(100)]
        return df
    
    def _create_fallback_handwriting_dataset(self):
        """Create minimal fallback handwriting dataset."""
        X, y = make_classification(n_samples=50, n_features=15, n_classes=2, random_state=123)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(15)])
        df['class'] = y
        df['subject_id'] = [f'hw_{i}' for i in range(50)]
        return df
    
    def _create_fallback_ppmi_dataset(self):
        """Create minimal fallback PPMI dataset."""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=789)
        df = pd.DataFrame(X, columns=[f'clinical_feature_{i}' for i in range(10)])
        df['group'] = ['PD' if label == 1 else 'Control' for label in y]
        df['subject_id'] = [f'ppmi_{i}' for i in range(200)]
        return df
    
    def load_voice_dataset(self):
        """
        Load voice dataset for analysis.
        
        Returns:
            pd.DataFrame: Voice dataset
        """
        return self.datasets.get('voice')
    
    def load_handwriting_dataset(self):
        """
        Load handwriting dataset for analysis.
        
        Returns:
            pd.DataFrame: Handwriting dataset
        """
        return self.datasets.get('handwriting')
    
    def load_ppmi_dataset(self):
        """
        Load PPMI clinical dataset for analysis.
        
        Returns:
            pd.DataFrame: PPMI dataset
        """
        return self.datasets.get('ppmi')
    
    def get_dataset_info(self, dataset_name):
        """
        Get information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset ('voice', 'handwriting', 'ppmi')
            
        Returns:
            dict: Dataset information
        """
        dataset_map = {
            'voice': 'uci_voice',
            'handwriting': 'handwriting',
            'ppmi': 'ppmi'
        }
        
        if dataset_name in dataset_map:
            key = dataset_map[dataset_name]
            info = self.dataset_info.get(key, {})
            
            # Add actual dataset statistics if loaded
            if dataset_name in self.datasets and self.datasets[dataset_name] is not None:
                df = self.datasets[dataset_name]
                info['loaded_samples'] = len(df)
                info['loaded_features'] = len(df.columns) - 2  # Exclude ID and target columns
                info['available'] = True
            else:
                info['available'] = False
            
            return info
        
        return None
    
    def get_available_datasets(self):
        """
        Get list of available datasets.
        
        Returns:
            list: Available dataset names
        """
        available = []
        for name in ['voice', 'handwriting', 'ppmi']:
            if name in self.datasets and self.datasets[name] is not None:
                available.append(name)
        
        return available
    
    def get_dataset_sample(self, dataset_name, n_samples=10):
        """
        Get a sample of the dataset for preview.
        
        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples to return
            
        Returns:
            pd.DataFrame: Sample of the dataset
        """
        if dataset_name in self.datasets and self.datasets[dataset_name] is not None:
            return self.datasets[dataset_name].head(n_samples)
        
        return None
