import numpy as np
import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
from scipy.signal import find_peaks, butter, filtfilt
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

class MotionProcessor:
    """
    Motion analysis processor for Parkinson's Disease detection.
    Analyzes hand movements, tremors, bradykinesia, and gait patterns.
    """
    
    def __init__(self):
        # Initialize MediaPipe Hands and Pose if available
        if mp is not None:
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.hands = None
            self.pose = None
            print("Warning: MediaPipe not available. Using basic motion analysis.")
        
        # Feature names for model input
        self.feature_names = [
            'tremor_frequency', 'tremor_amplitude', 'movement_speed',
            'movement_regularity', 'coordination_score', 'bradykinesia_score',
            'hand_stability', 'finger_agility', 'arm_swing_amplitude',
            'gait_velocity', 'step_length', 'postural_sway'
        ]
        
        # Key landmark indices for MediaPipe Hands
        self.hand_landmarks = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'thumb_mcp': 2,
            'index_mcp': 5,
            'middle_mcp': 9,
            'ring_mcp': 13,
            'pinky_mcp': 17
        }
        
        # Key pose landmarks for gait analysis
        self.pose_landmarks = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        # Thresholds for PD indicators
        self.pd_thresholds = {
            'tremor_frequency': (4.0, 6.0),  # Hz, PD range
            'movement_speed': 60,            # % of normal, below indicates bradykinesia
            'finger_tapping_rate': 4.0,      # taps/sec, below indicates bradykinesia
            'arm_swing': 20,                 # degrees, below indicates reduced swing
            'gait_speed': 1.0,               # m/s, below indicates slow gait
            'step_length': 55                # cm, below indicates shuffling
        }
    
    def analyze_tremor(self, hand_landmarks_sequence):
        """
        Analyze tremor characteristics from hand landmark sequence.
        
        Args:
            hand_landmarks_sequence: List of hand landmarks over time
            
        Returns:
            dict: Tremor analysis results
        """
        try:
            if not hand_landmarks_sequence or len(hand_landmarks_sequence) < 10:
                return self._default_tremor_analysis()
            
            # Extract wrist positions over time
            wrist_positions = []
            for landmarks in hand_landmarks_sequence:
                if landmarks and len(landmarks) > self.hand_landmarks['wrist']:
                    wrist = landmarks[self.hand_landmarks['wrist']]
                    wrist_positions.append([wrist.x, wrist.y])
            
            if len(wrist_positions) < 10:
                return self._default_tremor_analysis()
            
            wrist_positions = np.array(wrist_positions)
            
            # Calculate tremor frequency using FFT
            tremor_freq = self._calculate_tremor_frequency(wrist_positions)
            
            # Calculate tremor amplitude
            tremor_amplitude = self._calculate_tremor_amplitude(wrist_positions)
            
            # Assess tremor regularity
            tremor_regularity = self._assess_tremor_regularity(wrist_positions)
            
            # Determine tremor type (resting vs postural)
            tremor_type = self._classify_tremor_type(wrist_positions)
            
            return {
                'tremor_frequency': tremor_freq,
                'tremor_amplitude': tremor_amplitude,
                'tremor_regularity': tremor_regularity,
                'tremor_type': tremor_type,
                'pd_tremor_probability': self._calculate_pd_tremor_probability(tremor_freq, tremor_amplitude)
            }
            
        except Exception as e:
            print(f"Error analyzing tremor: {str(e)}")
            return self._default_tremor_analysis()
    
    def analyze_finger_tapping(self, hand_landmarks_sequence):
        """
        Analyze finger tapping patterns for bradykinesia assessment.
        
        Args:
            hand_landmarks_sequence: Sequence of hand landmarks during tapping
            
        Returns:
            dict: Finger tapping analysis results
        """
        try:
            if not hand_landmarks_sequence or len(hand_landmarks_sequence) < 20:
                return self._default_tapping_analysis()
            
            # Extract thumb and index finger positions
            thumb_positions = []
            index_positions = []
            
            for landmarks in hand_landmarks_sequence:
                if landmarks and len(landmarks) > max(self.hand_landmarks['thumb_tip'], self.hand_landmarks['index_tip']):
                    thumb = landmarks[self.hand_landmarks['thumb_tip']]
                    index = landmarks[self.hand_landmarks['index_tip']]
                    thumb_positions.append([thumb.x, thumb.y])
                    index_positions.append([index.x, index.y])
            
            if len(thumb_positions) < 20:
                return self._default_tapping_analysis()
            
            # Calculate distances between thumb and index finger
            distances = []
            for thumb, index in zip(thumb_positions, index_positions):
                dist = euclidean(thumb, index)
                distances.append(dist)
            
            # Detect tapping events (local minima in distance)
            tapping_events = self._detect_tapping_events(distances)
            
            # Calculate tapping metrics
            tapping_rate = len(tapping_events) / (len(distances) / 30)  # Assuming 30 FPS
            
            # Calculate amplitude consistency
            if len(tapping_events) > 2:
                tap_amplitudes = [distances[i] for i in tapping_events]
                amplitude_consistency = 1.0 - (np.std(tap_amplitudes) / np.mean(tap_amplitudes))
            else:
                amplitude_consistency = 0.5
            
            # Calculate rhythm regularity
            if len(tapping_events) > 3:
                intervals = np.diff(tapping_events)
                rhythm_regularity = 1.0 - (np.std(intervals) / np.mean(intervals))
            else:
                rhythm_regularity = 0.5
            
            # Assess fatigue (decline in performance over time)
            fatigue_score = self._assess_tapping_fatigue(distances, tapping_events)
            
            return {
                'tapping_rate': max(0, tapping_rate),
                'amplitude_consistency': max(0, min(1, amplitude_consistency)),
                'rhythm_regularity': max(0, min(1, rhythm_regularity)),
                'fatigue_score': max(0, min(1, fatigue_score)),
                'bradykinesia_score': self._calculate_bradykinesia_score(tapping_rate, amplitude_consistency, rhythm_regularity)
            }
            
        except Exception as e:
            print(f"Error analyzing finger tapping: {str(e)}")
            return self._default_tapping_analysis()
    
    def analyze_gait(self, pose_landmarks_sequence):
        """
        Analyze gait patterns from pose landmark sequence.
        
        Args:
            pose_landmarks_sequence: Sequence of pose landmarks during walking
            
        Returns:
            dict: Gait analysis results
        """
        try:
            if not pose_landmarks_sequence or len(pose_landmarks_sequence) < 30:
                return self._default_gait_analysis()
            
            # Extract key joint positions
            left_ankle_positions = []
            right_ankle_positions = []
            left_wrist_positions = []
            right_wrist_positions = []
            
            for landmarks in pose_landmarks_sequence:
                if landmarks and len(landmarks.landmark) > max(self.pose_landmarks.values()):
                    left_ankle = landmarks.landmark[self.pose_landmarks['left_ankle']]
                    right_ankle = landmarks.landmark[self.pose_landmarks['right_ankle']]
                    left_wrist = landmarks.landmark[self.pose_landmarks['left_wrist']]
                    right_wrist = landmarks.landmark[self.pose_landmarks['right_wrist']]
                    
                    left_ankle_positions.append([left_ankle.x, left_ankle.y])
                    right_ankle_positions.append([right_ankle.x, right_ankle.y])
                    left_wrist_positions.append([left_wrist.x, left_wrist.y])
                    right_wrist_positions.append([right_wrist.x, right_wrist.y])
            
            if len(left_ankle_positions) < 30:
                return self._default_gait_analysis()
            
            # Analyze step characteristics
            step_metrics = self._analyze_step_patterns(left_ankle_positions, right_ankle_positions)
            
            # Analyze arm swing
            arm_swing_metrics = self._analyze_arm_swing(left_wrist_positions, right_wrist_positions)
            
            # Calculate gait velocity
            gait_velocity = self._calculate_gait_velocity(left_ankle_positions, right_ankle_positions)
            
            # Assess gait symmetry
            gait_symmetry = self._assess_gait_symmetry(left_ankle_positions, right_ankle_positions)
            
            # Detect freezing episodes
            freezing_score = self._detect_freezing_episodes(left_ankle_positions, right_ankle_positions)
            
            return {
                'step_length': step_metrics['step_length'],
                'step_frequency': step_metrics['step_frequency'],
                'stride_variability': step_metrics['stride_variability'],
                'arm_swing_amplitude': arm_swing_metrics['amplitude'],
                'arm_swing_symmetry': arm_swing_metrics['symmetry'],
                'gait_velocity': gait_velocity,
                'gait_symmetry': gait_symmetry,
                'freezing_score': freezing_score,
                'pd_gait_probability': self._calculate_pd_gait_probability(step_metrics, arm_swing_metrics, gait_velocity)
            }
            
        except Exception as e:
            print(f"Error analyzing gait: {str(e)}")
            return self._default_gait_analysis()
    
    def _calculate_tremor_frequency(self, positions):
        """Calculate tremor frequency using FFT analysis."""
        try:
            # Calculate velocity (first derivative)
            velocity = np.diff(positions, axis=0)
            
            # Combine x and y components
            velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=1))
            
            # Apply FFT
            fft = np.fft.fft(velocity_magnitude)
            freqs = np.fft.fftfreq(len(velocity_magnitude), d=1/30)  # Assuming 30 FPS
            
            # Find dominant frequency in the 3-8 Hz range (tremor range)
            tremor_range = (freqs >= 3) & (freqs <= 8)
            if np.any(tremor_range):
                power_spectrum = np.abs(fft[tremor_range])
                dominant_freq_idx = np.argmax(power_spectrum)
                tremor_frequency = freqs[tremor_range][dominant_freq_idx]
                return abs(tremor_frequency)
            else:
                return 5.0  # Default value
                
        except Exception:
            return 5.0
    
    def _calculate_tremor_amplitude(self, positions):
        """Calculate tremor amplitude."""
        try:
            # Calculate position variance as amplitude measure
            x_var = np.var(positions[:, 0])
            y_var = np.var(positions[:, 1])
            amplitude = np.sqrt(x_var + y_var)
            
            # Normalize to 0-1 range
            return min(1.0, amplitude * 1000)  # Scale factor for normalization
            
        except Exception:
            return 0.3
    
    def _assess_tremor_regularity(self, positions):
        """Assess tremor regularity/consistency."""
        try:
            # Calculate autocorrelation to measure periodicity
            velocity = np.diff(positions, axis=0)
            velocity_magnitude = np.sqrt(np.sum(velocity**2, axis=1))
            
            # Remove mean and normalize
            velocity_centered = velocity_magnitude - np.mean(velocity_magnitude)
            if np.std(velocity_centered) > 0:
                velocity_normalized = velocity_centered / np.std(velocity_centered)
            else:
                return 0.5
            
            # Calculate autocorrelation
            autocorr = np.correlate(velocity_normalized, velocity_normalized, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = find_peaks(autocorr[1:], height=0.3)
            
            if len(peaks) > 0:
                # Regularity based on peak prominence
                max_peak = np.max(autocorr[peaks + 1])
                return min(1.0, max_peak)
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _classify_tremor_type(self, positions):
        """Classify tremor as resting or postural/action."""
        try:
            # Calculate movement magnitude
            velocity = np.diff(positions, axis=0)
            movement_magnitude = np.mean(np.sqrt(np.sum(velocity**2, axis=1)))
            
            # Resting tremor: high tremor with low overall movement
            # Postural/action tremor: tremor increases with movement
            
            if movement_magnitude < 0.01:  # Low movement threshold
                return "resting"
            else:
                return "postural/action"
                
        except Exception:
            return "unknown"
    
    def _calculate_pd_tremor_probability(self, frequency, amplitude):
        """Calculate probability that tremor is PD-related."""
        try:
            # PD tremor characteristics: 4-6 Hz, moderate amplitude
            freq_score = 1.0 if 4.0 <= frequency <= 6.0 else max(0, 1.0 - abs(frequency - 5.0) / 3.0)
            amp_score = 1.0 if 0.2 <= amplitude <= 0.8 else max(0, 1.0 - abs(amplitude - 0.5) / 0.5)
            
            return (freq_score + amp_score) / 2
            
        except Exception:
            return 0.5
    
    def _detect_tapping_events(self, distances):
        """Detect finger tapping events from distance signal."""
        try:
            # Smooth the signal
            if len(distances) > 5:
                kernel = np.ones(5) / 5
                smoothed = np.convolve(distances, kernel, mode='same')
            else:
                smoothed = distances
            
            # Find local minima (finger contact)
            peaks, _ = find_peaks(-np.array(smoothed), height=-np.mean(smoothed) * 0.8, distance=5)
            
            return peaks.tolist()
            
        except Exception:
            return []
    
    def _assess_tapping_fatigue(self, distances, tapping_events):
        """Assess fatigue during finger tapping test."""
        try:
            if len(tapping_events) < 4:
                return 0.2
            
            # Split into first and last halves
            mid_point = len(tapping_events) // 2
            first_half = tapping_events[:mid_point]
            second_half = tapping_events[mid_point:]
            
            # Calculate tapping rates for each half
            if len(first_half) > 1 and len(second_half) > 1:
                first_rate = len(first_half) / (first_half[-1] - first_half[0] + 1)
                second_rate = len(second_half) / (second_half[-1] - second_half[0] + 1)
                
                # Fatigue is the decline in rate
                fatigue = max(0, (first_rate - second_rate) / first_rate)
                return min(1.0, fatigue)
            else:
                return 0.2
                
        except Exception:
            return 0.2
    
    def _calculate_bradykinesia_score(self, tapping_rate, amplitude_consistency, rhythm_regularity):
        """Calculate overall bradykinesia score."""
        try:
            # Normal tapping rate is 4-6 taps per second
            rate_score = 1.0 if tapping_rate >= 4.0 else tapping_rate / 4.0
            
            # Combine metrics (lower score indicates more bradykinesia)
            overall_score = (rate_score + amplitude_consistency + rhythm_regularity) / 3
            
            # Invert to get bradykinesia score (higher = more bradykinesia)
            bradykinesia_score = 1.0 - overall_score
            
            return max(0, min(1, bradykinesia_score))
            
        except Exception:
            return 0.3
    
    def _analyze_step_patterns(self, left_ankle, right_ankle):
        """Analyze step patterns from ankle positions."""
        try:
            # Detect heel strikes (local minima in vertical position)
            left_y = [pos[1] for pos in left_ankle]
            right_y = [pos[1] for pos in right_ankle]
            
            # Find heel strikes
            left_strikes, _ = find_peaks(-np.array(left_y), distance=10)
            right_strikes, _ = find_peaks(-np.array(right_y), distance=10)
            
            # Calculate step metrics
            if len(left_strikes) > 2 and len(right_strikes) > 2:
                # Step frequency
                total_steps = len(left_strikes) + len(right_strikes)
                total_time = len(left_ankle) / 30  # Assuming 30 FPS
                step_frequency = total_steps / total_time if total_time > 0 else 2.0
                
                # Step length (approximate from stride pattern)
                left_stride_lengths = []
                for i in range(len(left_strikes) - 1):
                    stride_length = abs(left_ankle[left_strikes[i+1]][0] - left_ankle[left_strikes[i]][0])
                    left_stride_lengths.append(stride_length)
                
                if left_stride_lengths:
                    avg_step_length = np.mean(left_stride_lengths) * 100  # Convert to cm (approximate)
                    stride_variability = np.std(left_stride_lengths) / np.mean(left_stride_lengths) if np.mean(left_stride_lengths) > 0 else 0.1
                else:
                    avg_step_length = 60.0
                    stride_variability = 0.05
            else:
                step_frequency = 2.0
                avg_step_length = 60.0
                stride_variability = 0.05
            
            return {
                'step_frequency': step_frequency,
                'step_length': avg_step_length,
                'stride_variability': stride_variability
            }
            
        except Exception:
            return {
                'step_frequency': 2.0,
                'step_length': 60.0,
                'stride_variability': 0.05
            }
    
    def _analyze_arm_swing(self, left_wrist, right_wrist):
        """Analyze arm swing patterns."""
        try:
            # Calculate arm swing amplitude
            left_y = [pos[1] for pos in left_wrist]
            right_y = [pos[1] for pos in right_wrist]
            
            left_amplitude = max(left_y) - min(left_y) if len(left_y) > 0 else 0.1
            right_amplitude = max(right_y) - min(right_y) if len(right_y) > 0 else 0.1
            
            avg_amplitude = (left_amplitude + right_amplitude) / 2 * 180  # Convert to degrees (approximate)
            
            # Calculate symmetry
            if left_amplitude > 0 and right_amplitude > 0:
                symmetry = 1.0 - abs(left_amplitude - right_amplitude) / max(left_amplitude, right_amplitude)
            else:
                symmetry = 0.8
            
            return {
                'amplitude': min(50, max(10, avg_amplitude)),  # Clamp to reasonable range
                'symmetry': max(0, min(1, symmetry))
            }
            
        except Exception:
            return {
                'amplitude': 25.0,
                'symmetry': 0.8
            }
    
    def _calculate_gait_velocity(self, left_ankle, right_ankle):
        """Calculate gait velocity."""
        try:
            # Use horizontal movement of ankles to estimate velocity
            left_x = [pos[0] for pos in left_ankle]
            right_x = [pos[0] for pos in right_ankle]
            
            # Calculate average horizontal velocity
            left_velocity = np.mean(np.abs(np.diff(left_x))) * 30  # Convert to units/second
            right_velocity = np.mean(np.abs(np.diff(right_x))) * 30
            
            avg_velocity = (left_velocity + right_velocity) / 2
            
            # Convert to approximate m/s (this is a rough estimate)
            gait_velocity = avg_velocity * 2.0  # Scaling factor
            
            return min(2.0, max(0.5, gait_velocity))
            
        except Exception:
            return 1.2
    
    def _assess_gait_symmetry(self, left_ankle, right_ankle):
        """Assess gait symmetry between left and right legs."""
        try:
            # Compare step timing and amplitude between legs
            left_y = [pos[1] for pos in left_ankle]
            right_y = [pos[1] for pos in right_ankle]
            
            left_range = max(left_y) - min(left_y)
            right_range = max(right_y) - min(right_y)
            
            if left_range > 0 and right_range > 0:
                symmetry = 1.0 - abs(left_range - right_range) / max(left_range, right_range)
            else:
                symmetry = 0.8
            
            return max(0, min(1, symmetry))
            
        except Exception:
            return 0.8
    
    def _detect_freezing_episodes(self, left_ankle, right_ankle):
        """Detect freezing of gait episodes."""
        try:
            # Look for periods of very low movement
            all_positions = left_ankle + right_ankle
            
            # Calculate movement magnitude over time
            movement_magnitudes = []
            for i in range(len(all_positions) - 1):
                mag = euclidean(all_positions[i], all_positions[i+1])
                movement_magnitudes.append(mag)
            
            # Find periods of very low movement
            threshold = np.mean(movement_magnitudes) * 0.2
            freezing_frames = sum(1 for mag in movement_magnitudes if mag < threshold)
            
            # Convert to freezing score (0 = no freezing, 1 = severe freezing)
            freezing_score = freezing_frames / len(movement_magnitudes) if movement_magnitudes else 0
            
            return min(1.0, freezing_score)
            
        except Exception:
            return 0.1
    
    def _calculate_pd_gait_probability(self, step_metrics, arm_swing_metrics, gait_velocity):
        """Calculate probability that gait pattern is PD-related."""
        try:
            risk_score = 0
            
            # Check step length
            if step_metrics['step_length'] < 55:
                risk_score += 1
            
            # Check arm swing
            if arm_swing_metrics['amplitude'] < 20:
                risk_score += 1
            
            # Check gait velocity
            if gait_velocity < 1.0:
                risk_score += 1
            
            # Check stride variability
            if step_metrics['stride_variability'] > 0.05:
                risk_score += 1
            
            # Convert to probability
            return min(1.0, risk_score / 4.0)
            
        except Exception:
            return 0.3
    
    def _default_tremor_analysis(self):
        """Return default tremor analysis when calculation fails."""
        return {
            'tremor_frequency': 5.0,
            'tremor_amplitude': 0.3,
            'tremor_regularity': 0.6,
            'tremor_type': 'unknown',
            'pd_tremor_probability': 0.4
        }
    
    def _default_tapping_analysis(self):
        """Return default tapping analysis when calculation fails."""
        return {
            'tapping_rate': 4.5,
            'amplitude_consistency': 0.7,
            'rhythm_regularity': 0.8,
            'fatigue_score': 0.2,
            'bradykinesia_score': 0.3
        }
    
    def _default_gait_analysis(self):
        """Return default gait analysis when calculation fails."""
        return {
            'step_length': 60.0,
            'step_frequency': 2.0,
            'stride_variability': 0.05,
            'arm_swing_amplitude': 25.0,
            'arm_swing_symmetry': 0.8,
            'gait_velocity': 1.2,
            'gait_symmetry': 0.8,
            'freezing_score': 0.1,
            'pd_gait_probability': 0.3
        }
    
    def predict_parkinsons_motion(self, features, model=None):
        """
        Make Parkinson's prediction using motion features.
        
        Args:
            features: dict of extracted motion features
            model: trained model object (optional)
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            if model is not None:
                # Use provided model
                feature_vector = self._features_to_vector(features)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba([feature_vector])[0]
                    prediction = 1 if probabilities[1] > 0.5 else 0
                    confidence = probabilities[1] * 100
                else:
                    prediction = model.predict([feature_vector])[0]
                    confidence = 80.0 if prediction == 1 else 20.0
                
                return int(prediction), float(confidence)
            
            else:
                # Rule-based prediction
                risk_score = 0
                
                # Check tremor characteristics
                tremor_freq = features.get('tremor_frequency', 5.0)
                if 4.0 <= tremor_freq <= 6.0:
                    risk_score += 2
                
                # Check bradykinesia
                if features.get('bradykinesia_score', 0.3) > 0.5:
                    risk_score += 2
                
                # Check gait abnormalities
                if features.get('gait_velocity', 1.2) < 1.0:
                    risk_score += 1
                
                if features.get('arm_swing_amplitude', 25.0) < 20:
                    risk_score += 1
                
                if features.get('step_length', 60.0) < 55:
                    risk_score += 1
                
                prediction = 1 if risk_score >= 3 else 0
                confidence = min(95, max(55, risk_score * 15 + 25))
                
                return prediction, confidence
                
        except Exception as e:
            print(f"Error making motion prediction: {str(e)}")
            return 0, 50.0
    
    def _features_to_vector(self, features):
        """Convert features dictionary to numpy vector for model input."""
        # Define expected feature order for model
        expected_features = [
            'tremor_frequency', 'tremor_amplitude', 'movement_speed',
            'movement_regularity', 'coordination_score', 'bradykinesia_score',
            'hand_stability', 'finger_agility', 'arm_swing_amplitude',
            'gait_velocity', 'step_length', 'postural_sway'
        ]
        
        # Create feature vector
        feature_vector = []
        for feature_name in expected_features:
            value = features.get(feature_name, 0.5)  # Default value if missing
            feature_vector.append(float(value))
        
        return np.array(feature_vector)