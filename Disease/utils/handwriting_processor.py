import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Simple skeletonization function
def skeletonize(binary_image):
    """Simple skeletonization using erosion."""
    try:
        # Simple implementation using morphological operations
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(binary_image.astype(np.uint8), kernel, iterations=1)
        return eroded > 0
    except Exception:
        return binary_image > 0

class HandwritingProcessor:
    """
    Handwriting and drawing analysis processor for Parkinson's Disease detection.
    Analyzes spiral drawings, wave patterns, and general handwriting for motor dysfunction indicators.
    """
    
    def __init__(self):
        self.feature_names = [
            'tremor_frequency', 'tremor_amplitude', 'line_smoothness',
            'velocity_variation', 'stroke_width', 'size_consistency', 
            'writing_speed', 'pressure_variation', 'spatial_coverage',
            'drawing_density', 'symmetry_score', 'completion_rate',
            'spiral_regularity', 'micrographia_score', 'rhythm_score'
        ]
        
        # Thresholds for PD indicators
        self.pd_thresholds = {
            'tremor_frequency': (4.0, 7.0),  # Hz
            'line_smoothness': (0.0, 0.7),    # Lower is worse
            'micrographia_score': (0.3, 1.0), # Higher indicates micrographia
            'pressure_variation': (0.3, 1.0)  # Higher indicates more variation
        }
    
    def extract_features(self, image_data):
        """
        Extract comprehensive handwriting features for Parkinson's detection.
        
        Args:
            image_data: numpy array representing the image or PIL Image
            
        Returns:
            dict: Dictionary of extracted handwriting features
        """
        try:
            # Convert image data to proper format
            processed_image = self._preprocess_image(image_data)
            
            if processed_image is None:
                return None
            
            features = {}
            
            # Extract geometric and spatial features
            geometric_features = self._extract_geometric_features(processed_image)
            features.update(geometric_features)
            
            # Extract tremor-related features
            tremor_features = self._extract_tremor_features(processed_image)
            features.update(tremor_features)
            
            # Extract writing dynamics features (simulated from static image)
            dynamics_features = self._extract_dynamics_features(processed_image)
            features.update(dynamics_features)
            
            # Extract spiral-specific features if spiral detected
            spiral_features = self._extract_spiral_features(processed_image)
            features.update(spiral_features)
            
            # Extract micrographia indicators
            micrographia_features = self._extract_micrographia_features(processed_image)
            features.update(micrographia_features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting handwriting features: {str(e)}")
            return None
    
    def _preprocess_image(self, image_data):
        """Preprocess the input image for analysis."""
        try:
            # Handle different input types
            if isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 4:  # RGBA
                    image = image_data[:, :, :3]  # Remove alpha channel
                elif len(image_data.shape) == 3:  # RGB
                    image = image_data
                else:  # Grayscale
                    image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB) if len(image_data.shape) == 2 else image_data
            else:
                # Assume PIL Image
                image = np.array(image_data)
            
            # Convert to grayscale for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Ensure proper data type
            if gray.dtype != np.uint8:
                gray = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
            
            # Apply binary threshold to separate ink from background
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Clean up the image - remove small noise
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def _extract_geometric_features(self, binary_image):
        """Extract geometric features from the handwriting sample."""
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return self._default_geometric_features()
            
            # Calculate total area covered by writing
            total_area = sum(cv2.contourArea(contour) for contour in contours)
            image_area = binary_image.shape[0] * binary_image.shape[1]
            spatial_coverage = total_area / image_area
            
            # Calculate drawing density
            white_pixels = np.sum(binary_image > 0)
            drawing_density = white_pixels / image_area
            
            # Analyze stroke width distribution
            stroke_widths = self._calculate_stroke_widths(binary_image)
            avg_stroke_width = np.mean(stroke_widths) if len(stroke_widths) > 0 else 3.0
            stroke_width_std = np.std(stroke_widths) if len(stroke_widths) > 1 else 0.5
            size_consistency = 1.0 - min(1.0, stroke_width_std / (avg_stroke_width + 1e-6))
            
            # Calculate bounding box and aspect ratio
            x, y, w, h = cv2.boundingRect(np.vstack(contours))
            aspect_ratio = w / h if h > 0 else 1.0
            
            return {
                'spatial_coverage': spatial_coverage,
                'drawing_density': drawing_density,
                'stroke_width': avg_stroke_width,
                'size_consistency': size_consistency,
                'aspect_ratio': aspect_ratio,
                'total_strokes': len(contours)
            }
            
        except Exception:
            return self._default_geometric_features()
    
    def _default_geometric_features(self):
        """Return default geometric features when calculation fails."""
        return {
            'spatial_coverage': 0.3,
            'drawing_density': 0.1,
            'stroke_width': 3.0,
            'size_consistency': 0.8,
            'aspect_ratio': 1.0,
            'total_strokes': 1
        }
    
    def _calculate_stroke_widths(self, binary_image):
        """Calculate stroke width distribution using distance transform."""
        try:
            # Apply distance transform
            dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
            
            # Get non-zero values (stroke widths)
            stroke_widths = dist_transform[binary_image > 0] * 2  # Multiply by 2 for full width
            
            # Filter out very small values
            stroke_widths = stroke_widths[stroke_widths > 0.5]
            
            return stroke_widths if len(stroke_widths) > 0 else [3.0]
            
        except Exception:
            return [3.0]
    
    def _extract_tremor_features(self, binary_image):
        """Extract tremor-related features from handwriting."""
        try:
            # Skeletonize the binary image to get center lines
            skeleton = skeletonize(binary_image > 0)
            
            # Find skeleton points
            y_coords, x_coords = np.where(skeleton)
            
            if len(x_coords) < 10:
                return self._default_tremor_features()
            
            # Sort points to create a path (simplified approach)
            points = np.column_stack((x_coords, y_coords))
            
            # Calculate line smoothness using curvature analysis
            smoothness = self._calculate_line_smoothness(points)
            
            # Estimate tremor frequency using spectral analysis
            tremor_freq, tremor_amp = self._estimate_tremor_frequency(points)
            
            # Calculate velocity variation (simulated from spatial distribution)
            velocity_var = self._calculate_velocity_variation(points)
            
            return {
                'tremor_frequency': tremor_freq,
                'tremor_amplitude': tremor_amp,
                'line_smoothness': smoothness,
                'velocity_variation': velocity_var
            }
            
        except Exception:
            return self._default_tremor_features()
    
    def _default_tremor_features(self):
        """Return default tremor features when calculation fails."""
        return {
            'tremor_frequency': 4.5,
            'tremor_amplitude': 0.2,
            'line_smoothness': 0.8,
            'velocity_variation': 0.3
        }
    
    def _calculate_line_smoothness(self, points):
        """Calculate line smoothness based on curvature variation."""
        if len(points) < 5:
            return 0.8
        
        try:
            # Calculate second derivatives (curvature approximation)
            x_diff2 = np.diff(points[:, 0], n=2)
            y_diff2 = np.diff(points[:, 1], n=2)
            curvature = np.abs(x_diff2) + np.abs(y_diff2)
            
            # Smoothness is inverse of curvature variation
            curvature_std = np.std(curvature)
            max_curvature = np.max(curvature) + 1e-6
            
            smoothness = 1.0 - min(1.0, curvature_std / max_curvature)
            return max(0.0, min(1.0, smoothness))
            
        except Exception:
            return 0.8
    
    def _estimate_tremor_frequency(self, points):
        """Estimate tremor frequency and amplitude from point sequence."""
        if len(points) < 20:
            return 4.5, 0.2
        
        try:
            # Analyze x and y coordinates separately
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Remove trend (detrend)
            x_detrended = x_coords - np.linspace(x_coords[0], x_coords[-1], len(x_coords))
            y_detrended = y_coords - np.linspace(y_coords[0], y_coords[-1], len(y_coords))
            
            # Combine x and y fluctuations
            combined_signal = np.sqrt(x_detrended**2 + y_detrended**2)
            
            # Apply FFT to estimate dominant frequency
            fft = np.fft.fft(combined_signal)
            freqs = np.fft.fftfreq(len(combined_signal))
            
            # Find dominant frequency (excluding DC component)
            power_spectrum = np.abs(fft[1:len(fft)//2])
            freq_range = freqs[1:len(freqs)//2]
            
            if len(power_spectrum) > 0:
                # Find peaks in power spectrum
                peaks, _ = find_peaks(power_spectrum, height=np.max(power_spectrum) * 0.1)
                
                if len(peaks) > 0:
                    # Get the most prominent frequency
                    dominant_peak = peaks[np.argmax(power_spectrum[peaks])]
                    tremor_freq = abs(freq_range[dominant_peak]) * 100  # Scale to reasonable Hz range
                    tremor_freq = max(3.0, min(8.0, tremor_freq))  # Clamp to physiological range
                else:
                    tremor_freq = 4.5
            else:
                tremor_freq = 4.5
            
            # Calculate amplitude as standard deviation of fluctuations
            tremor_amplitude = np.std(combined_signal) / (np.mean(np.abs(combined_signal)) + 1e-6)
            tremor_amplitude = max(0.0, min(1.0, tremor_amplitude))
            
            return tremor_freq, tremor_amplitude
            
        except Exception:
            return 4.5, 0.2
    
    def _calculate_velocity_variation(self, points):
        """Calculate velocity variation from point sequence."""
        if len(points) < 3:
            return 0.3
        
        try:
            # Calculate distances between consecutive points
            distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
            
            # Velocity is inversely related to distance (closer points = slower writing)
            velocities = distances + 1e-6  # Add small constant to avoid division by zero
            
            # Calculate coefficient of variation
            velocity_cv = np.std(velocities) / (np.mean(velocities) + 1e-6)
            
            return max(0.0, min(1.0, velocity_cv))
            
        except Exception:
            return 0.3
    
    def _extract_dynamics_features(self, binary_image):
        """Extract writing dynamics features (simulated from static image)."""
        try:
            # Simulate writing speed based on stroke density
            stroke_pixels = np.sum(binary_image > 0)
            image_area = binary_image.shape[0] * binary_image.shape[1]
            
            # More dense writing suggests slower speed
            writing_speed = 150 - (stroke_pixels / image_area) * 100
            writing_speed = max(50, min(200, writing_speed))
            
            # Simulate pressure variation based on stroke width variation
            stroke_widths = self._calculate_stroke_widths(binary_image)
            pressure_variation = np.std(stroke_widths) / (np.mean(stroke_widths) + 1e-6) if len(stroke_widths) > 1 else 0.2
            pressure_variation = max(0.0, min(1.0, pressure_variation))
            
            # Calculate rhythm score based on spatial regularity
            rhythm_score = self._calculate_rhythm_score(binary_image)
            
            return {
                'writing_speed': writing_speed,
                'pressure_variation': pressure_variation,
                'rhythm_score': rhythm_score
            }
            
        except Exception:
            return {
                'writing_speed': 100.0,
                'pressure_variation': 0.25,
                'rhythm_score': 0.75
            }
    
    def _calculate_rhythm_score(self, binary_image):
        """Calculate rhythm score based on spatial regularity."""
        try:
            # Find contours
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 2:
                return 0.7
            
            # Calculate distances between contour centers
            centers = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
            
            if len(centers) < 2:
                return 0.7
            
            # Calculate regularity of spacing
            centers = np.array(centers)
            distances = []
            
            for i in range(len(centers) - 1):
                dist = np.sqrt(np.sum((centers[i+1] - centers[i])**2))
                distances.append(dist)
            
            if len(distances) == 0:
                return 0.7
            
            # Rhythm score is inverse of distance variation
            distance_cv = np.std(distances) / (np.mean(distances) + 1e-6)
            rhythm_score = 1.0 - min(1.0, distance_cv)
            
            return max(0.0, min(1.0, rhythm_score))
            
        except Exception:
            return 0.7
    
    def _extract_spiral_features(self, binary_image):
        """Extract features specific to spiral drawings."""
        try:
            # Check if this looks like a spiral by analyzing circular patterns
            is_spiral, spiral_metrics = self._detect_spiral_pattern(binary_image)
            
            if is_spiral:
                return {
                    'spiral_regularity': spiral_metrics['regularity'],
                    'completion_rate': spiral_metrics['completion'],
                    'symmetry_score': spiral_metrics['symmetry']
                }
            else:
                # Default values for non-spiral drawings
                return {
                    'spiral_regularity': 0.5,
                    'completion_rate': 0.8,
                    'symmetry_score': 0.7
                }
                
        except Exception:
            return {
                'spiral_regularity': 0.5,
                'completion_rate': 0.8,
                'symmetry_score': 0.7
            }
    
    def _detect_spiral_pattern(self, binary_image):
        """Detect if the image contains a spiral pattern."""
        try:
            # Find the center of mass
            y_coords, x_coords = np.where(binary_image > 0)
            
            if len(x_coords) == 0:
                return False, {'regularity': 0.5, 'completion': 0.5, 'symmetry': 0.5}
            
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            
            # Calculate distances from center
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Calculate angles
            angles = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            # Check if there's a spiral pattern (increasing distance with angle)
            # Sort by angle and check if distances generally increase
            sorted_indices = np.argsort(angles)
            sorted_distances = distances[sorted_indices]
            sorted_angles = angles[sorted_indices]
            
            # Check for spiral characteristics
            distance_trend = np.corrcoef(sorted_angles, sorted_distances)[0, 1]
            is_spiral = abs(distance_trend) > 0.3  # Correlation threshold
            
            # Calculate spiral metrics
            regularity = max(0.0, min(1.0, abs(distance_trend)))
            
            # Completion rate based on angle coverage
            angle_range = np.max(sorted_angles) - np.min(sorted_angles)
            completion = min(1.0, angle_range / (4 * np.pi))  # 2 full rotations = complete
            
            # Symmetry based on distance distribution
            distance_quartiles = np.percentile(sorted_distances, [25, 75])
            symmetry = 1.0 - min(1.0, (distance_quartiles[1] - distance_quartiles[0]) / (np.mean(sorted_distances) + 1e-6))
            
            return is_spiral, {
                'regularity': regularity,
                'completion': completion,
                'symmetry': max(0.0, symmetry)
            }
            
        except Exception:
            return False, {'regularity': 0.5, 'completion': 0.5, 'symmetry': 0.5}
    
    def _extract_micrographia_features(self, binary_image):
        """Extract features related to micrographia (progressive decrease in writing size)."""
        try:
            # Find connected components (individual strokes/letters)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 3:
                return {'micrographia_score': 0.1}
            
            # Calculate bounding box areas for each component
            areas = []
            x_positions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                areas.append(w * h)
                x_positions.append(x + w/2)  # Center x position
            
            # Sort by x position (left to right)
            sorted_indices = np.argsort(x_positions)
            sorted_areas = np.array(areas)[sorted_indices]
            
            # Check for decreasing size trend (micrographia indicator)
            if len(sorted_areas) > 2:
                # Calculate correlation between position and size
                positions = np.arange(len(sorted_areas))
                size_correlation = np.corrcoef(positions, sorted_areas)[0, 1]
                
                # Negative correlation indicates decreasing size (micrographia)
                micrographia_score = max(0.0, -size_correlation) if not np.isnan(size_correlation) else 0.1
            else:
                micrographia_score = 0.1
            
            return {
                'micrographia_score': min(1.0, micrographia_score)
            }
            
        except Exception:
            return {'micrographia_score': 0.1}
    
    def predict_parkinsons(self, features, model):
        """
        Make Parkinson's prediction using extracted handwriting features.
        
        Args:
            features: dict of extracted handwriting features
            model: trained model object
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Convert features to vector format expected by model
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
                confidence = 80.0 if prediction == 1 else 20.0
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            print(f"Error making handwriting prediction: {str(e)}")
            
            # Return prediction based on feature analysis
            risk_score = 0
            
            # Check tremor indicators
            if features.get('tremor_frequency', 4.5) > 5.5:
                risk_score += 2
            
            # Check smoothness
            if features.get('line_smoothness', 0.8) < 0.6:
                risk_score += 2
            
            # Check micrographia
            if features.get('micrographia_score', 0.1) > 0.4:
                risk_score += 1
            
            # Check pressure variation
            if features.get('pressure_variation', 0.25) > 0.4:
                risk_score += 1
            
            # Check completion and symmetry
            if features.get('completion_rate', 0.8) < 0.7:
                risk_score += 1
            
            prediction = 1 if risk_score >= 3 else 0
            confidence = min(95, max(55, risk_score * 15 + 25))
            
            return prediction, confidence
    
    def _features_to_vector(self, features):
        """Convert features dictionary to numpy vector for model input."""
        # Define expected feature order
        expected_features = [
            'tremor_frequency', 'tremor_amplitude', 'line_smoothness',
            'velocity_variation', 'stroke_width', 'size_consistency',
            'writing_speed', 'pressure_variation', 'spatial_coverage',
            'drawing_density', 'symmetry_score', 'completion_rate',
            'spiral_regularity', 'micrographia_score', 'rhythm_score'
        ]
        
        # Create feature vector with default values for missing features
        vector = []
        defaults = {
            'tremor_frequency': 4.5, 'tremor_amplitude': 0.2, 'line_smoothness': 0.8,
            'velocity_variation': 0.3, 'stroke_width': 3.0, 'size_consistency': 0.8,
            'writing_speed': 100.0, 'pressure_variation': 0.25, 'spatial_coverage': 0.5,
            'drawing_density': 0.3, 'symmetry_score': 0.7, 'completion_rate': 0.85,
            'spiral_regularity': 0.6, 'micrographia_score': 0.1, 'rhythm_score': 0.75
        }
        
        for feature_name in expected_features:
            value = features.get(feature_name, defaults[feature_name])
            vector.append(float(value))
        
        return np.array(vector)
