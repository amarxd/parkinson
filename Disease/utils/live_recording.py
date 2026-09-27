import cv2
import numpy as np
import tempfile
import threading
import queue
import time
from datetime import datetime
import os

# Try to import sounddevice, but handle gracefully if not available
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except (ImportError, OSError):
    AUDIO_AVAILABLE = False
    sd = None
    sf = None

class LiveVideoRecorder:
    """
    Live video recording using OpenCV for facial analysis.
    """
    
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=100)
        
    def start_recording(self, duration_seconds=30, camera_index=0):
        """
        Start live video recording from webcam.
        
        Args:
            duration_seconds: Duration to record in seconds
            camera_index: Camera device index (0 for default)
            
        Returns:
            tuple: (success, frames_array, message)
        """
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                return False, None, "Camera not accessible in this environment. This feature requires direct hardware access which is not available in cloud/web deployments. Please use the 'Upload Photo' option instead."
            
            self.is_recording = True
            self.frames = []
            
            start_time = time.time()
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = 0
            max_frames = int(duration_seconds * fps)
            
            while self.is_recording and (time.time() - start_time) < duration_seconds:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Convert BGR to RGB for processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(rgb_frame)
                frame_count += 1
                
                # Put frame in queue for real-time display
                if not self.frame_queue.full():
                    self.frame_queue.put(rgb_frame)
                
                # Limit frame count to prevent memory issues
                if frame_count >= max_frames:
                    break
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
            
            self.is_recording = False
            self.cap.release()
            
            if len(self.frames) == 0:
                return False, None, "No frames captured. Camera may be in use by another application."
            
            frames_array = np.array(self.frames)
            return True, frames_array, f"Successfully recorded {len(self.frames)} frames"
            
        except Exception as e:
            self.is_recording = False
            if self.cap:
                self.cap.release()
            return False, None, f"Error during recording: {str(e)}"
    
    def stop_recording(self):
        """Stop the recording."""
        self.is_recording = False
        if self.cap:
            self.cap.release()
    
    def get_latest_frame(self):
        """Get the latest frame from the queue."""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def save_recording(self, output_path=None, fps=30):
        """
        Save recorded frames as video file.
        
        Args:
            output_path: Path to save video (default: temp file)
            fps: Frames per second for output video
            
        Returns:
            str: Path to saved video file
        """
        if not self.frames:
            return None
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/facial_recording_{timestamp}.avi"
        
        try:
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in self.frames:
                # Convert RGB back to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            return output_path
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            return None


class LiveAudioRecorder:
    """
    Live audio recording using sounddevice for voice analysis.
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.is_recording = False
        self.audio_data = []
        self.channels = 1
        
    def record_audio(self, duration_seconds=5):
        """
        Record audio from microphone.
        
        Args:
            duration_seconds: Duration to record in seconds
            
        Returns:
            tuple: (success, audio_array, sample_rate, message)
        """
        # Check if audio is available in this environment
        if not AUDIO_AVAILABLE or sd is None:
            return False, None, None, "Audio recording not available in this environment. PortAudio library is required but not found. This feature works on systems with direct microphone access."
        
        try:
            # Check if audio device is available
            devices = sd.query_devices()
            if not devices:
                return False, None, None, "No audio devices found"
            
            # Find default input device
            default_input = sd.query_devices(kind='input')
            if not default_input:
                return False, None, None, "No audio input device available"
            
            self.is_recording = True
            self.audio_data = []
            
            # Record audio
            print(f"Recording audio for {duration_seconds} seconds...")
            recording = sd.rec(
                int(duration_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            
            # Wait for recording to complete
            sd.wait()
            
            self.is_recording = False
            self.audio_data = recording.flatten()
            
            if len(self.audio_data) == 0:
                return False, None, None, "No audio data captured"
            
            return True, self.audio_data, self.sample_rate, f"Successfully recorded {duration_seconds}s of audio"
            
        except Exception as e:
            self.is_recording = False
            return False, None, None, f"Error recording audio: {str(e)}"
    
    def save_audio(self, audio_data, output_path=None):
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data array
            output_path: Path to save audio (default: temp file)
            
        Returns:
            str: Path to saved audio file
        """
        if not AUDIO_AVAILABLE or sf is None:
            print("Warning: soundfile not available, skipping audio save")
            return None
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use current directory instead of /tmp for better compatibility
            output_path = f"voice_recording_{timestamp}.wav"
        
        try:
            # Ensure audio_data is the correct type
            if audio_data is not None and len(audio_data) > 0:
                sf.write(output_path, audio_data, self.sample_rate)
                return output_path
            else:
                print("Warning: No audio data to save")
                return None
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            # Try alternative path
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='.') as tmp:
                    sf.write(tmp.name, audio_data, self.sample_rate)
                    return tmp.name
            except:
                return None
    
    def get_audio_info(self, audio_data):
        """
        Get information about recorded audio.
        
        Args:
            audio_data: Audio data array
            
        Returns:
            dict: Audio information
        """
        try:
            duration = len(audio_data) / self.sample_rate
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data**2))
            
            return {
                'duration': duration,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'samples': len(audio_data),
                'max_amplitude': max_amplitude,
                'rms_level': rms,
                'peak_db': 20 * np.log10(max_amplitude) if max_amplitude > 0 else -np.inf
            }
        except Exception as e:
            print(f"Error getting audio info: {str(e)}")
            return {}


class MotionDataCollector:
    """
    Collect user-entered motion detection report details.
    """
    
    def __init__(self):
        self.symptoms = {
            'tremor': {
                'present': False,
                'severity': 'None',
                'location': [],
                'frequency': 'Never',
                'notes': ''
            },
            'bradykinesia': {
                'present': False,
                'severity': 'None',
                'affected_activities': [],
                'notes': ''
            },
            'rigidity': {
                'present': False,
                'severity': 'None',
                'location': [],
                'notes': ''
            },
            'postural_instability': {
                'present': False,
                'severity': 'None',
                'fall_history': False,
                'notes': ''
            },
            'gait_disturbance': {
                'present': False,
                'type': [],
                'severity': 'None',
                'notes': ''
            }
        }
        
    def collect_tremor_data(self, present, severity='None', location=None, frequency='Never', notes=''):
        """Collect tremor symptom data."""
        self.symptoms['tremor'] = {
            'present': present,
            'severity': severity,
            'location': location or [],
            'frequency': frequency,
            'notes': notes
        }
        
    def collect_bradykinesia_data(self, present, severity='None', affected_activities=None, notes=''):
        """Collect bradykinesia symptom data."""
        self.symptoms['bradykinesia'] = {
            'present': present,
            'severity': severity,
            'affected_activities': affected_activities or [],
            'notes': notes
        }
        
    def collect_rigidity_data(self, present, severity='None', location=None, notes=''):
        """Collect rigidity symptom data."""
        self.symptoms['rigidity'] = {
            'present': present,
            'severity': severity,
            'location': location or [],
            'notes': notes
        }
        
    def collect_postural_data(self, present, severity='None', fall_history=False, notes=''):
        """Collect postural instability data."""
        self.symptoms['postural_instability'] = {
            'present': present,
            'severity': severity,
            'fall_history': fall_history,
            'notes': notes
        }
        
    def collect_gait_data(self, present, gait_type=None, severity='None', notes=''):
        """Collect gait disturbance data."""
        self.symptoms['gait_disturbance'] = {
            'present': present,
            'type': gait_type or [],
            'severity': severity,
            'notes': notes
        }
        
    def calculate_motor_score(self):
        """
        Calculate overall motor symptom score.
        
        Returns:
            dict: Motor scores and risk assessment
        """
        severity_scores = {
            'None': 0,
            'Mild': 1,
            'Moderate': 2,
            'Severe': 3,
            'Very Severe': 4
        }
        
        total_score = 0
        symptom_count = 0
        
        for symptom_name, data in self.symptoms.items():
            if data.get('present', False):
                symptom_count += 1
                severity = data.get('severity', 'None')
                total_score += severity_scores.get(severity, 0)
        
        max_possible_score = len(self.symptoms) * 4
        normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0
        
        # Determine risk level
        if normalized_score >= 0.6:
            risk_level = 'High'
        elif normalized_score >= 0.3:
            risk_level = 'Moderate'
        elif symptom_count > 0:
            risk_level = 'Low'
        else:
            risk_level = 'Minimal'
        
        return {
            'total_score': total_score,
            'max_score': max_possible_score,
            'normalized_score': normalized_score,
            'symptom_count': symptom_count,
            'risk_level': risk_level,
            'symptoms_present': [name for name, data in self.symptoms.items() if data.get('present', False)]
        }
    
    def get_summary(self):
        """Get a summary of collected data."""
        motor_scores = self.calculate_motor_score()
        
        summary = {
            'motor_scores': motor_scores,
            'detailed_symptoms': self.symptoms,
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
