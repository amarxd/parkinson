import numpy as np
import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

class FacialProcessor:
    """
    Facial analysis processor for Parkinson's Disease detection.
    Analyzes facial expressions, symmetry, and facial masking indicators.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh if available
        if mp is not None:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        else:
            self.face_mesh = None
            print("Warning: MediaPipe not available. Using basic CV analysis.")
        
        # Feature names for model input
        self.feature_names = [
            'expression_mobility', 'eye_expression', 'mouth_mobility',
            'forehead_expression', 'facial_symmetry', 'left_right_balance',
            'eye_symmetry', 'mouth_symmetry', 'blink_frequency',
            'landmarks_quality', 'face_orientation', 'expression_range'
        ]
        
        # Key landmark indices for MediaPipe Face Mesh
        self.key_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'nose': [1, 2, 5, 4, 6, 19, 94, 125, 141, 235, 31, 228, 229, 230, 231, 232],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'jaw': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
        
        # Thresholds for PD indicators
        self.pd_thresholds = {
            'expression_mobility': 0.5,  # Below this indicates reduced expression
            'facial_symmetry': 0.8,      # Below this indicates asymmetry
            'blink_frequency': 15,       # Below this per minute indicates reduced blinking
            'mouth_mobility': 0.4        # Below this indicates reduced mouth movement
        }
    
    def extract_facial_features(self, image_data):
        """
        Extract comprehensive facial features for Parkinson's detection.
        
        Args:
            image_data: numpy array representing the facial image
            
        Returns:
            dict: Dictionary of extracted facial features
        """
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image_data)
            
            if processed_image is None:
                return None
            
            features = {}
            
            # Extract MediaPipe-based features if available
            if self.face_mesh is not None:
                landmark_features = self._extract_landmark_features(processed_image)
                features.update(landmark_features)
            
            # Extract traditional CV features as backup
            cv_features = self._extract_cv_features(processed_image)
            features.update(cv_features)
            
            # Calculate composite features
            composite_features = self._calculate_composite_features(features)
            features.update(composite_features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting facial features: {str(e)}")
            return self._default_features()
    
    def _preprocess_image(self, image_data):
        """Preprocess the input image for facial analysis."""
        try:
            # Handle different input types
            if isinstance(image_data, np.ndarray):
                if len(image_data.shape) == 4:  # RGBA
                    image = image_data[:, :, :3]  # Remove alpha channel
                elif len(image_data.shape) == 3:  # RGB
                    image = image_data
                else:  # Grayscale
                    image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
            else:
                # Assume PIL Image
                image = np.array(image_data)
            
            # Convert to RGB if BGR
            if image.shape[2] == 3:
                # Assume it's RGB already for MediaPipe
                rgb_image = image
            
            # Ensure proper data type
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1 else rgb_image.astype(np.uint8)
            
            return rgb_image
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None
    
    def _extract_landmark_features(self, rgb_image):
        """Extract features using MediaPipe Face Mesh."""
        try:
            # Process the image with MediaPipe
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return self._default_landmark_features()
            
            # Get the first detected face
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
            
            # Scale landmarks to image dimensions
            h, w = rgb_image.shape[:2]
            landmarks[:, 0] *= w
            landmarks[:, 1] *= h
            
            features = {}
            
            # Extract facial symmetry
            features.update(self._calculate_facial_symmetry(landmarks))
            
            # Extract expression mobility
            features.update(self._calculate_expression_mobility(landmarks))
            
            # Extract eye-specific features
            features.update(self._calculate_eye_features(landmarks))
            
            # Extract mouth-specific features
            features.update(self._calculate_mouth_features(landmarks))
            
            # Calculate landmark quality
            features['landmarks_quality'] = self._calculate_landmark_quality(landmarks)
            features['landmarks_count'] = len(landmarks)
            
            # Calculate face orientation
            features['face_orientation'] = self._calculate_face_orientation(landmarks)
            
            return features
            
        except Exception as e:
            print(f"Error extracting landmark features: {str(e)}")
            return self._default_landmark_features()
    
    def _calculate_facial_symmetry(self, landmarks):
        """Calculate facial symmetry metrics."""
        try:
            # Get face center line
            nose_tip = landmarks[1]  # Nose tip
            forehead_center = landmarks[9]  # Forehead center
            chin_center = landmarks[175]  # Chin center
            
            # Calculate face midline
            midline_x = (nose_tip[0] + forehead_center[0] + chin_center[0]) / 3
            
            # Calculate symmetry for different facial regions
            left_eye_landmarks = [landmarks[i] for i in self.key_landmarks['left_eye'] if i < len(landmarks)]
            right_eye_landmarks = [landmarks[i] for i in self.key_landmarks['right_eye'] if i < len(landmarks)]
            
            # Eye symmetry
            if left_eye_landmarks and right_eye_landmarks:
                left_eye_center = np.mean(left_eye_landmarks, axis=0)
                right_eye_center = np.mean(right_eye_landmarks, axis=0)
                
                left_dist = abs(left_eye_center[0] - midline_x)
                right_dist = abs(right_eye_center[0] - midline_x)
                eye_symmetry = 1.0 - abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)
            else:
                eye_symmetry = 0.85
            
            # Mouth symmetry
            mouth_landmarks = [landmarks[i] for i in self.key_landmarks['mouth'] if i < len(landmarks)]
            if mouth_landmarks:
                mouth_left = min(mouth_landmarks, key=lambda x: x[0])
                mouth_right = max(mouth_landmarks, key=lambda x: x[0])
                mouth_center = (mouth_left[0] + mouth_right[0]) / 2
                mouth_symmetry = 1.0 - abs(mouth_center - midline_x) / (abs(mouth_center) + abs(midline_x) + 1e-6)
            else:
                mouth_symmetry = 0.85
            
            # Overall facial symmetry
            facial_symmetry = (eye_symmetry + mouth_symmetry) / 2
            
            return {
                'facial_symmetry': max(0.0, min(1.0, facial_symmetry)),
                'eye_symmetry': max(0.0, min(1.0, eye_symmetry)),
                'mouth_symmetry': max(0.0, min(1.0, mouth_symmetry)),
                'left_right_balance': max(0.0, min(1.0, (eye_symmetry + mouth_symmetry) / 2))
            }
            
        except Exception:
            return {
                'facial_symmetry': 0.85,
                'eye_symmetry': 0.85,
                'mouth_symmetry': 0.85,
                'left_right_balance': 0.85
            }
    
    def _calculate_expression_mobility(self, landmarks):
        """Calculate expression mobility indicators."""
        try:
            # Calculate various expression-related distances
            
            # Eye openness (vertical distance between eyelids)
            left_eye_landmarks = [landmarks[i] for i in self.key_landmarks['left_eye'] if i < len(landmarks)]
            right_eye_landmarks = [landmarks[i] for i in self.key_landmarks['right_eye'] if i < len(landmarks)]
            
            if left_eye_landmarks and right_eye_landmarks:
                # Approximate eye openness
                left_eye_height = max([lm[1] for lm in left_eye_landmarks]) - min([lm[1] for lm in left_eye_landmarks])
                right_eye_height = max([lm[1] for lm in right_eye_landmarks]) - min([lm[1] for lm in right_eye_landmarks])
                avg_eye_height = (left_eye_height + right_eye_height) / 2
                
                # Normalize eye openness (typical range: 5-20 pixels)
                eye_expression = min(1.0, avg_eye_height / 15.0)
            else:
                eye_expression = 0.6
            
            # Mouth mobility (width and height)
            mouth_landmarks = [landmarks[i] for i in self.key_landmarks['mouth'] if i < len(landmarks)]
            if mouth_landmarks:
                mouth_width = max([lm[0] for lm in mouth_landmarks]) - min([lm[0] for lm in mouth_landmarks])
                mouth_height = max([lm[1] for lm in mouth_landmarks]) - min([lm[1] for lm in mouth_landmarks])
                
                # Normalize mouth mobility
                mouth_mobility = min(1.0, (mouth_width + mouth_height) / 40.0)
            else:
                mouth_mobility = 0.5
            
            # Forehead expression (eyebrow position relative to eyes)
            eyebrow_landmarks = [landmarks[i] for i in self.key_landmarks['eyebrows'] if i < len(landmarks)]
            if eyebrow_landmarks and left_eye_landmarks:
                eyebrow_center = np.mean(eyebrow_landmarks, axis=0)
                eye_center = np.mean(left_eye_landmarks + right_eye_landmarks, axis=0)
                eyebrow_eye_distance = abs(eyebrow_center[1] - eye_center[1])
                
                # Normalize forehead expression
                forehead_expression = min(1.0, eyebrow_eye_distance / 30.0)
            else:
                forehead_expression = 0.4
            
            # Overall expression mobility
            expression_mobility = (eye_expression + mouth_mobility + forehead_expression) / 3
            
            return {
                'expression_mobility': max(0.0, min(1.0, expression_mobility)),
                'eye_expression': max(0.0, min(1.0, eye_expression)),
                'mouth_mobility': max(0.0, min(1.0, mouth_mobility)),
                'forehead_expression': max(0.0, min(1.0, forehead_expression))
            }
            
        except Exception:
            return {
                'expression_mobility': 0.6,
                'eye_expression': 0.6,
                'mouth_mobility': 0.5,
                'forehead_expression': 0.4
            }
    
    def _calculate_eye_features(self, landmarks):
        """Calculate eye-specific features."""
        try:
            left_eye_landmarks = [landmarks[i] for i in self.key_landmarks['left_eye'] if i < len(landmarks)]
            right_eye_landmarks = [landmarks[i] for i in self.key_landmarks['right_eye'] if i < len(landmarks)]
            
            if not (left_eye_landmarks and right_eye_landmarks):
                return {'eye_aspect_ratio': 0.3, 'eye_distance': 60.0}
            
            # Calculate Eye Aspect Ratio (EAR) for blink detection
            def eye_aspect_ratio(eye_landmarks):
                if len(eye_landmarks) < 6:
                    return 0.3
                
                # Vertical distances
                A = euclidean(eye_landmarks[1], eye_landmarks[5])
                B = euclidean(eye_landmarks[2], eye_landmarks[4])
                
                # Horizontal distance
                C = euclidean(eye_landmarks[0], eye_landmarks[3])
                
                # EAR calculation
                ear = (A + B) / (2.0 * C) if C > 0 else 0.3
                return ear
            
            left_ear = eye_aspect_ratio(left_eye_landmarks[:6])
            right_ear = eye_aspect_ratio(right_eye_landmarks[:6])
            avg_ear = (left_ear + right_ear) / 2
            
            # Calculate inter-eye distance
            left_eye_center = np.mean(left_eye_landmarks, axis=0)
            right_eye_center = np.mean(right_eye_landmarks, axis=0)
            eye_distance = euclidean(left_eye_center, right_eye_center)
            
            return {
                'eye_aspect_ratio': avg_ear,
                'eye_distance': eye_distance,
                'left_eye_ear': left_ear,
                'right_eye_ear': right_ear
            }
            
        except Exception:
            return {
                'eye_aspect_ratio': 0.3,
                'eye_distance': 60.0,
                'left_eye_ear': 0.3,
                'right_eye_ear': 0.3
            }
    
    def _calculate_mouth_features(self, landmarks):
        """Calculate mouth-specific features."""
        try:
            mouth_landmarks = [landmarks[i] for i in self.key_landmarks['mouth'] if i < len(landmarks)]
            
            if len(mouth_landmarks) < 4:
                return {'mouth_aspect_ratio': 0.5, 'mouth_width': 40.0}
            
            # Calculate mouth aspect ratio
            mouth_points = np.array(mouth_landmarks)
            mouth_width = max(mouth_points[:, 0]) - min(mouth_points[:, 0])
            mouth_height = max(mouth_points[:, 1]) - min(mouth_points[:, 1])
            
            mouth_aspect_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.5
            
            # Calculate mouth curvature (smile/frown detection)
            if len(mouth_landmarks) >= 6:
                left_corner = mouth_landmarks[0]
                right_corner = mouth_landmarks[3]
                center_top = mouth_landmarks[1]
                center_bottom = mouth_landmarks[4]
                
                corners_y = (left_corner[1] + right_corner[1]) / 2
                center_y = (center_top[1] + center_bottom[1]) / 2
                
                # Positive values indicate smile, negative indicate frown
                mouth_curvature = corners_y - center_y
            else:
                mouth_curvature = 0.0
            
            return {
                'mouth_aspect_ratio': mouth_aspect_ratio,
                'mouth_width': mouth_width,
                'mouth_height': mouth_height,
                'mouth_curvature': mouth_curvature
            }
            
        except Exception:
            return {
                'mouth_aspect_ratio': 0.5,
                'mouth_width': 40.0,
                'mouth_height': 20.0,
                'mouth_curvature': 0.0
            }
    
    def _calculate_landmark_quality(self, landmarks):
        """Calculate the quality of landmark detection."""
        try:
            # Check if we have the expected number of landmarks
            expected_landmarks = 468  # MediaPipe Face Mesh
            detection_ratio = len(landmarks) / expected_landmarks
            
            # Check landmark distribution
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            
            # Calculate the spread of landmarks
            x_spread = (np.max(x_coords) - np.min(x_coords)) / np.mean([np.max(x_coords), np.min(x_coords)])
            y_spread = (np.max(y_coords) - np.min(y_coords)) / np.mean([np.max(y_coords), np.min(y_coords)])
            
            # Quality is based on detection ratio and landmark spread
            spread_score = min(1.0, (x_spread + y_spread) / 2)
            quality = (detection_ratio + spread_score) / 2
            
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.8
    
    def _calculate_face_orientation(self, landmarks):
        """Calculate face orientation angle."""
        try:
            # Use nose tip and nose bridge to estimate face angle
            nose_tip = landmarks[1]
            nose_bridge = landmarks[9]
            
            # Calculate angle from vertical
            dx = nose_tip[0] - nose_bridge[0]
            dy = nose_tip[1] - nose_bridge[1]
            
            angle = np.arctan2(dx, dy) * 180 / np.pi
            
            return angle
            
        except Exception:
            return 0.0
    
    def _extract_cv_features(self, rgb_image):
        """Extract features using traditional computer vision methods."""
        try:
            # Convert to grayscale for traditional CV
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Face detection using Haar cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            features = {}
            
            if len(faces) > 0:
                # Get the largest face
                face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Calculate basic metrics
                features['face_size'] = w * h
                features['face_width'] = w
                features['face_height'] = h
                features['face_aspect_ratio'] = w / h if h > 0 else 1.0
                
                # Analyze texture (roughness indicator)
                features['face_texture'] = self._calculate_texture_features(face_roi)
                
            else:
                # Default values when no face detected
                features = {
                    'face_size': 15000,
                    'face_width': 150,
                    'face_height': 200,
                    'face_aspect_ratio': 0.75,
                    'face_texture': 0.5
                }
            
            return features
            
        except Exception:
            return {
                'face_size': 15000,
                'face_width': 150,
                'face_height': 200,
                'face_aspect_ratio': 0.75,
                'face_texture': 0.5
            }
    
    def _calculate_texture_features(self, face_roi):
        """Calculate texture features from face region."""
        try:
            # Calculate local variance as texture measure
            kernel = np.ones((5, 5), np.float32) / 25
            face_blur = cv2.filter2D(face_roi.astype(np.float32), -1, kernel)
            variance = np.var(face_roi.astype(np.float32) - face_blur)
            
            # Normalize variance
            texture_score = min(1.0, variance / 1000.0)
            
            return texture_score
            
        except Exception:
            return 0.5
    
    def _calculate_composite_features(self, features):
        """Calculate composite features from basic measurements."""
        try:
            # Calculate expression range (variability indicator)
            expression_components = [
                features.get('eye_expression', 0.6),
                features.get('mouth_mobility', 0.5),
                features.get('forehead_expression', 0.4)
            ]
            expression_range = max(expression_components) - min(expression_components)
            
            # Calculate blink frequency estimate (based on eye aspect ratio)
            ear = features.get('eye_aspect_ratio', 0.3)
            # Lower EAR suggests more closed eyes, potentially indicating reduced blinking
            estimated_blink_freq = 20 - (0.3 - ear) * 50 if ear < 0.3 else 20
            estimated_blink_freq = max(5, min(30, estimated_blink_freq))
            
            return {
                'expression_range': expression_range,
                'blink_frequency': estimated_blink_freq
            }
            
        except Exception:
            return {
                'expression_range': 0.2,
                'blink_frequency': 18
            }
    
    def _default_landmark_features(self):
        """Return default landmark features when detection fails."""
        return {
            'facial_symmetry': 0.85,
            'eye_symmetry': 0.85,
            'mouth_symmetry': 0.85,
            'left_right_balance': 0.85,
            'expression_mobility': 0.6,
            'eye_expression': 0.6,
            'mouth_mobility': 0.5,
            'forehead_expression': 0.4,
            'landmarks_quality': 0.7,
            'landmarks_count': 300,
            'face_orientation': 0.0,
            'eye_aspect_ratio': 0.3,
            'eye_distance': 60.0
        }
    
    def _default_features(self):
        """Return default features when extraction fails completely."""
        features = self._default_landmark_features()
        features.update({
            'face_size': 15000,
            'face_width': 150,
            'face_height': 200,
            'face_aspect_ratio': 0.75,
            'face_texture': 0.5,
            'expression_range': 0.2,
            'blink_frequency': 18
        })
        return features
    
    def predict_parkinsons_facial(self, features, model=None):
        """
        Make Parkinson's prediction using facial features.
        
        Args:
            features: dict of extracted facial features
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
                
                # Check expression mobility
                if features.get('expression_mobility', 0.6) < 0.5:
                    risk_score += 2
                
                # Check facial symmetry
                if features.get('facial_symmetry', 0.85) < 0.8:
                    risk_score += 1
                
                # Check blink frequency
                if features.get('blink_frequency', 18) < 15:
                    risk_score += 2
                
                # Check mouth mobility
                if features.get('mouth_mobility', 0.5) < 0.4:
                    risk_score += 1
                
                # Check eye expression
                if features.get('eye_expression', 0.6) < 0.4:
                    risk_score += 1
                
                prediction = 1 if risk_score >= 3 else 0
                confidence = min(95, max(55, risk_score * 15 + 25))
                
                return prediction, confidence
                
        except Exception as e:
            print(f"Error making facial prediction: {str(e)}")
            return 0, 50.0
    
    def _features_to_vector(self, features):
        """Convert features dictionary to numpy vector for model input."""
        # Define expected feature order for model
        expected_features = [
            'expression_mobility', 'eye_expression', 'mouth_mobility',
            'forehead_expression', 'facial_symmetry', 'left_right_balance',
            'eye_symmetry', 'mouth_symmetry', 'blink_frequency',
            'landmarks_quality', 'face_orientation', 'expression_range'
        ]
        
        # Create feature vector
        feature_vector = []
        for feature_name in expected_features:
            value = features.get(feature_name, 0.5)  # Default value if missing
            feature_vector.append(float(value))
        
        return np.array(feature_vector)