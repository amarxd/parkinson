import numpy as np
import pandas as pd
from scipy.signal import find_peaks, stft
import warnings
warnings.filterwarnings('ignore')

class VoiceProcessor:
    """
    Voice processing class for extracting Parkinson's disease biomarkers from speech signals.
    Based on research from UCI Oxford Parkinson Dataset and clinical literature.
    """
    
    def __init__(self):
        self.sample_rate = 22050
        self.frame_length = 2048
        self.hop_length = 512
        
        # Standard voice features for PD detection
        self.feature_names = [
            'jitter_rap', 'jitter_ppq5', 'jitter_ddp',
            'shimmer', 'shimmer_apq3', 'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda',
            'nhr', 'hnr', 'rpde', 'dfa', 'ppe',
            'f0_mean', 'f0_std', 'f0_min', 'f0_max',
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'mfcc_mean', 'mfcc_std'
        ]
    
    def extract_features(self, audio_data, sr=None):
        """
        Extract comprehensive voice features for Parkinson's detection.
        
        Args:
            audio_data: numpy array of audio signal
            sr: sample rate (optional, defaults to self.sample_rate)
        
        Returns:
            dict: Dictionary of extracted voice features
        """
        try:
            if sr is None:
                sr = self.sample_rate
            
            # Ensure audio is mono and properly formatted
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            features = {}
            
            # Extract fundamental frequency (F0) using librosa
            f0 = self._extract_f0(audio_data, sr)
            
            # Jitter measurements (frequency perturbation)
            jitter_features = self._calculate_jitter(f0)
            features.update(jitter_features)
            
            # Shimmer measurements (amplitude perturbation)  
            shimmer_features = self._calculate_shimmer(audio_data, sr)
            features.update(shimmer_features)
            
            # Noise measurements
            noise_features = self._calculate_noise_measures(audio_data, sr, f0)
            features.update(noise_features)
            
            # Nonlinear dynamical complexity measures
            complexity_features = self._calculate_complexity_measures(f0)
            features.update(complexity_features)
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio_data, sr)
            features.update(spectral_features)
            
            # MFCC features
            mfcc_features = self._extract_mfcc_features(audio_data, sr)
            features.update(mfcc_features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting voice features: {str(e)}")
            return None
    
    def _extract_f0(self, audio_data, sr):
        """Extract fundamental frequency using autocorrelation method."""
        return self._f0_zero_crossing(audio_data, sr)
    
    def _f0_zero_crossing(self, audio_data, sr):
        """Fallback F0 extraction using zero-crossing rate."""
        # Simple autocorrelation-based F0 estimation
        window_size = int(0.04 * sr)  # 40ms windows
        hop_size = int(0.01 * sr)     # 10ms hop
        
        f0_values = []
        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i:i + window_size]
            
            # Autocorrelation
            autocorr = np.correlate(window, window, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr[20:], height=0.3 * np.max(autocorr))
            
            if len(peaks) > 0:
                # Convert lag to frequency
                f0 = sr / (peaks[0] + 20)
                f0_values.append(f0 if 50 <= f0 <= 400 else np.nan)
            else:
                f0_values.append(np.nan)
        
        return np.array(f0_values)
    
    def _calculate_jitter(self, f0):
        """Calculate jitter (frequency perturbation) measures."""
        # Remove NaN values
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) < 3:
            return {
                'jitter_rap': 0.0,
                'jitter_ppq5': 0.0, 
                'jitter_ddp': 0.0,
                'f0_mean': 0.0,
                'f0_std': 0.0,
                'f0_min': 0.0,
                'f0_max': 0.0
            }
        
        # Period values (inverse of frequency)
        periods = 1.0 / (valid_f0 + 1e-8)
        
        # RAP (Relative Average Perturbation)
        if len(periods) >= 3:
            rap_sum = 0
            for i in range(1, len(periods) - 1):
                rap_sum += abs(periods[i] - (periods[i-1] + periods[i] + periods[i+1]) / 3)
            jitter_rap = (rap_sum / (len(periods) - 2)) / np.mean(periods) if len(periods) > 2 else 0.0
        else:
            jitter_rap = 0.0
        
        # PPQ5 (5-point Period Perturbation Quotient)
        if len(periods) >= 5:
            ppq5_sum = 0
            for i in range(2, len(periods) - 2):
                local_mean = np.mean(periods[i-2:i+3])
                ppq5_sum += abs(periods[i] - local_mean)
            jitter_ppq5 = (ppq5_sum / (len(periods) - 4)) / np.mean(periods) if len(periods) > 4 else 0.0
        else:
            jitter_ppq5 = 0.0
        
        # DDP (Difference of Differences of Periods)
        if len(periods) >= 3:
            ddp_sum = 0
            for i in range(2, len(periods)):
                ddp_sum += abs(periods[i] - periods[i-1] - (periods[i-1] - periods[i-2]))
            jitter_ddp = ddp_sum / (len(periods) - 2) / np.mean(periods) if len(periods) > 2 else 0.0
        else:
            jitter_ddp = 0.0
        
        return {
            'jitter_rap': jitter_rap,
            'jitter_ppq5': jitter_ppq5,
            'jitter_ddp': jitter_ddp,
            'f0_mean': np.mean(valid_f0),
            'f0_std': np.std(valid_f0),
            'f0_min': np.min(valid_f0),
            'f0_max': np.max(valid_f0)
        }
    
    def _calculate_shimmer(self, audio_data, sr):
        """Calculate shimmer (amplitude perturbation) measures."""
        # Extract amplitude envelope using scipy STFT
        _, _, Zxx = stft(audio_data, sr, nperseg=self.frame_length, noverlap=self.frame_length-self.hop_length)
        amplitude = np.abs(Zxx)
        amplitude_mean = np.mean(amplitude, axis=0)
        
        if len(amplitude_mean) < 3:
            return {
                'shimmer': 0.0,
                'shimmer_apq3': 0.0,
                'shimmer_apq5': 0.0,
                'shimmer_apq11': 0.0,
                'shimmer_dda': 0.0
            }
        
        # Basic shimmer (local)
        shimmer_local = 0
        for i in range(1, len(amplitude_mean)):
            shimmer_local += abs(amplitude_mean[i] - amplitude_mean[i-1])
        shimmer = (shimmer_local / (len(amplitude_mean) - 1)) / np.mean(amplitude_mean) if len(amplitude_mean) > 1 else 0.0
        
        # APQ3 (3-point Amplitude Perturbation Quotient)
        if len(amplitude_mean) >= 3:
            apq3_sum = 0
            for i in range(1, len(amplitude_mean) - 1):
                local_mean = (amplitude_mean[i-1] + amplitude_mean[i] + amplitude_mean[i+1]) / 3
                apq3_sum += abs(amplitude_mean[i] - local_mean)
            shimmer_apq3 = (apq3_sum / (len(amplitude_mean) - 2)) / np.mean(amplitude_mean)
        else:
            shimmer_apq3 = 0.0
        
        # APQ5 (5-point Amplitude Perturbation Quotient)
        if len(amplitude_mean) >= 5:
            apq5_sum = 0
            for i in range(2, len(amplitude_mean) - 2):
                local_mean = np.mean(amplitude_mean[i-2:i+3])
                apq5_sum += abs(amplitude_mean[i] - local_mean)
            shimmer_apq5 = (apq5_sum / (len(amplitude_mean) - 4)) / np.mean(amplitude_mean)
        else:
            shimmer_apq5 = 0.0
        
        # APQ11 (11-point Amplitude Perturbation Quotient)  
        if len(amplitude_mean) >= 11:
            apq11_sum = 0
            for i in range(5, len(amplitude_mean) - 5):
                local_mean = np.mean(amplitude_mean[i-5:i+6])
                apq11_sum += abs(amplitude_mean[i] - local_mean)
            shimmer_apq11 = (apq11_sum / (len(amplitude_mean) - 10)) / np.mean(amplitude_mean)
        else:
            shimmer_apq11 = 0.0
        
        # DDA (Difference of Differences of Amplitudes)
        if len(amplitude_mean) >= 3:
            dda_sum = 0
            for i in range(2, len(amplitude_mean)):
                dda_sum += abs(amplitude_mean[i] - amplitude_mean[i-1] - (amplitude_mean[i-1] - amplitude_mean[i-2]))
            shimmer_dda = dda_sum / (len(amplitude_mean) - 2) / np.mean(amplitude_mean)
        else:
            shimmer_dda = 0.0
        
        return {
            'shimmer': shimmer,
            'shimmer_apq3': shimmer_apq3,
            'shimmer_apq5': shimmer_apq5,
            'shimmer_apq11': shimmer_apq11,
            'shimmer_dda': shimmer_dda
        }
    
    def _calculate_noise_measures(self, audio_data, sr, f0):
        """Calculate noise-related measures (NHR, HNR)."""
        try:
            # Noise-to-Harmonics Ratio (NHR)
            # Simplified implementation using spectral analysis
            _, _, Zxx = stft(audio_data, sr, nperseg=self.frame_length, noverlap=self.frame_length-self.hop_length)
            magnitude = np.abs(Zxx)
            
            # Estimate harmonic and noise components
            harmonic_power = np.sum(magnitude[:magnitude.shape[0]//4, :], axis=0)
            total_power = np.sum(magnitude, axis=0)
            noise_power = total_power - harmonic_power
            
            nhr = np.mean(noise_power / (harmonic_power + 1e-8))
            
            # Harmonics-to-Noise Ratio (HNR) 
            hnr = 10 * np.log10(np.mean(harmonic_power / (noise_power + 1e-8)) + 1e-8)
            
            return {
                'nhr': max(0, min(nhr, 1.0)),  # Clamp to reasonable range
                'hnr': max(-10, min(hnr, 40))   # Clamp HNR to typical range
            }
            
        except Exception:
            return {'nhr': 0.1, 'hnr': 20.0}  # Default values
    
    def _calculate_complexity_measures(self, f0):
        """Calculate nonlinear dynamical complexity measures (RPDE, DFA, PPE)."""
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) < 10:
            return {'rpde': 0.5, 'dfa': 0.7, 'ppe': 0.2}
        
        try:
            # RPDE (Recurrence Period Density Entropy) - simplified
            rpde = self._calculate_rpde_simple(valid_f0)
            
            # DFA (Detrended Fluctuation Analysis) - simplified
            dfa = self._calculate_dfa_simple(valid_f0)
            
            # PPE (Pitch Period Entropy) - simplified
            ppe = self._calculate_ppe_simple(valid_f0)
            
            return {
                'rpde': max(0, min(rpde, 1.0)),
                'dfa': max(0, min(dfa, 2.0)), 
                'ppe': max(0, min(ppe, 1.0))
            }
            
        except Exception:
            return {'rpde': 0.5, 'dfa': 0.7, 'ppe': 0.2}
    
    def _calculate_rpde_simple(self, signal):
        """Simplified RPDE calculation."""
        # Basic entropy calculation of period distribution
        periods = 1.0 / (signal + 1e-8)
        hist, _ = np.histogram(periods, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero entries
        return -np.sum(hist * np.log2(hist + 1e-8)) / np.log2(len(hist))
    
    def _calculate_dfa_simple(self, signal):
        """Simplified DFA calculation."""
        # Detrended fluctuation analysis approximation
        N = len(signal)
        if N < 10:
            return 0.7
            
        # Integrate the signal
        y = np.cumsum(signal - np.mean(signal))
        
        # Calculate fluctuations for different window sizes
        window_sizes = [4, 8, 16, 32]
        fluctuations = []
        
        for window_size in window_sizes:
            if window_size >= N // 2:
                continue
                
            # Divide into non-overlapping windows
            n_windows = N // window_size
            F = 0
            
            for i in range(n_windows):
                start = i * window_size
                end = start + window_size
                window_y = y[start:end]
                
                # Linear detrend
                x = np.arange(window_size)
                coeffs = np.polyfit(x, window_y, 1)
                trend = np.polyval(coeffs, x)
                
                # Calculate fluctuation
                F += np.mean((window_y - trend) ** 2)
            
            fluctuations.append(F / n_windows)
        
        if len(fluctuations) < 2:
            return 0.7
        
        # Calculate scaling exponent
        log_windows = np.log2(window_sizes[:len(fluctuations)])
        log_fluct = np.log2(fluctuations)
        
        if len(log_windows) > 1:
            slope, _ = np.polyfit(log_windows, log_fluct, 1)
            return abs(slope)
        
        return 0.7
    
    def _calculate_ppe_simple(self, f0):
        """Simplified PPE calculation."""
        periods = 1.0 / (f0 + 1e-8)
        
        # Calculate relative period differences
        if len(periods) < 2:
            return 0.2
            
        diff = np.abs(np.diff(periods))
        relative_diff = diff / (periods[:-1] + 1e-8)
        
        # Entropy of relative differences
        hist, _ = np.histogram(relative_diff, bins=10, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.2
            
        return -np.sum(hist * np.log2(hist + 1e-8)) / np.log2(len(hist))
    
    def _extract_spectral_features(self, audio_data, sr):
        """Extract spectral features using scipy."""
        try:
            # Get STFT
            _, _, Zxx = stft(audio_data, sr, nperseg=self.frame_length, noverlap=self.frame_length-self.hop_length)
            magnitude = np.abs(Zxx)
            
            # Frequency bins
            freqs = np.linspace(0, sr/2, magnitude.shape[0])
            
            # Spectral centroid
            spectral_centroids = []
            for frame in range(magnitude.shape[1]):
                power = magnitude[:, frame] ** 2
                if np.sum(power) > 0:
                    centroid = np.sum(freqs * power) / np.sum(power)
                    spectral_centroids.append(centroid)
                else:
                    spectral_centroids.append(1000.0)
            
            # Spectral rolloff (85% energy)
            spectral_rolloff_vals = []
            for frame in range(magnitude.shape[1]):
                power = magnitude[:, frame] ** 2
                total_power = np.sum(power)
                if total_power > 0:
                    cumsum_power = np.cumsum(power)
                    rolloff_idx = np.where(cumsum_power >= 0.85 * total_power)[0]
                    if len(rolloff_idx) > 0:
                        spectral_rolloff_vals.append(freqs[rolloff_idx[0]])
                    else:
                        spectral_rolloff_vals.append(sr/4)
                else:
                    spectral_rolloff_vals.append(sr/4)
            
            # Spectral bandwidth
            spectral_bandwidth_vals = []
            for frame in range(magnitude.shape[1]):
                power = magnitude[:, frame] ** 2
                if np.sum(power) > 0:
                    centroid = np.sum(freqs * power) / np.sum(power)
                    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power))
                    spectral_bandwidth_vals.append(spread)
                else:
                    spectral_bandwidth_vals.append(1500.0)
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
            zcr = len(zero_crossings) / len(audio_data)
            
            return {
                'spectral_centroid': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff_vals),
                'spectral_bandwidth': np.mean(spectral_bandwidth_vals),
                'zero_crossing_rate': zcr
            }
            
        except Exception:
            return {
                'spectral_centroid': 1000.0,
                'spectral_rolloff': 2000.0,
                'spectral_bandwidth': 1500.0,
                'zero_crossing_rate': 0.1
            }
    
    def _extract_mfcc_features(self, audio_data, sr):
        """Extract simplified MFCC-like features."""
        try:
            # Get STFT for basic spectral analysis
            _, _, Zxx = stft(audio_data, sr, nperseg=self.frame_length, noverlap=self.frame_length-self.hop_length)
            magnitude = np.abs(Zxx)
            
            # Simplified mel-scale approximation
            # Use log of spectral energy in different frequency bands
            n_bands = 13
            band_energies = []
            
            for band in range(n_bands):
                start_idx = int(band * magnitude.shape[0] / n_bands)
                end_idx = int((band + 1) * magnitude.shape[0] / n_bands)
                band_energy = np.mean(magnitude[start_idx:end_idx, :], axis=0)
                log_energy = np.log(band_energy + 1e-8)
                band_energies.append(log_energy)
            
            # Convert to numpy array
            mfcc_like = np.array(band_energies)
            
            return {
                'mfcc_mean': np.mean(mfcc_like),
                'mfcc_std': np.std(mfcc_like)
            }
            
        except Exception:
            return {
                'mfcc_mean': -5.0,
                'mfcc_std': 15.0
            }
    
    def predict_parkinsons(self, features, model):
        """
        Make Parkinson's prediction using extracted features and loaded model.
        
        Args:
            features: dict of extracted voice features
            model: trained model object
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Convert features to array format expected by model
            feature_vector = self._features_to_vector(features)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                # Model supports probability prediction
                probabilities = model.predict_proba([feature_vector])[0]
                prediction = 1 if probabilities[1] > 0.5 else 0
                confidence = probabilities[1] * 100
            else:
                # Simple binary prediction
                prediction = model.predict([feature_vector])[0]
                confidence = 75.0 if prediction == 1 else 25.0
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            # Return default prediction based on features
            risk_indicators = 0
            if features.get('jitter_rap', 0) > 0.005:
                risk_indicators += 1
            if features.get('shimmer', 0) > 0.04:
                risk_indicators += 1
            if features.get('nhr', 0) > 0.15:
                risk_indicators += 1
            if features.get('hnr', 30) < 20:
                risk_indicators += 1
                
            prediction = 1 if risk_indicators >= 2 else 0
            confidence = min(95, max(55, risk_indicators * 20 + 15))
            
            return prediction, confidence
    
    def _features_to_vector(self, features):
        """Convert features dictionary to numpy vector for model input."""
        # Define the expected order of features for the model
        expected_features = [
            'jitter_rap', 'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_apq3',
            'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 
            'rpde', 'dfa', 'ppe', 'f0_mean', 'f0_std', 'spectral_centroid',
            'spectral_rolloff', 'spectral_bandwidth', 'zero_crossing_rate',
            'mfcc_mean', 'mfcc_std'
        ]
        
        # Extract features in the correct order, with defaults for missing features
        vector = []
        for feature_name in expected_features:
            value = features.get(feature_name, 0.0)
            vector.append(float(value))
        
        return np.array(vector)
