import cv2
import numpy as np
import os
import time
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque

class HealthDetector:
    def __init__(self, buffer_size=300, fps=30):
        """Initialize the health detector with buffer size and FPS."""
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Time and base signals
        self.start_time = None
        self.times = []
        
        # Pulse detection variables
        self.green_values = []
        self.pulse_history = deque(maxlen=10)
        self.stable_pulse = None
        self.pulse_stable_count = 0
        
        # SpO2 detection variables
        self.red_values = []
        self.blue_values = []
        self.spo2_history = deque(maxlen=10)
        self.stable_spo2 = None
        self.spo2_stable_count = 0
        
        # Signal quality
        self.signal_quality = 0
        
        # Signal processing components
        self.red_ac_values = deque(maxlen=10)
        self.blue_ac_values = deque(maxlen=10)
        self.red_dc_values = deque(maxlen=10)
        self.blue_dc_values = deque(maxlen=10)
        
        # Face detection setup
        self.face_cascade = self._load_face_cascade()
    
    def _load_face_cascade(self):
        """Load Haar cascade for face detection."""
        try:
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                'C:/opencv/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        return cascade
            
            print("Face cascade not found. Face detection disabled.")
            return None
        except Exception as e:
            print(f"Could not load face cascade: {e}")
            return None
    
    def butter_bandpass(self, lowcut=0.8, highcut=3.0, order=3):
        """Create a bandpass filter for signal processing."""
        nyquist = self.fps / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def detect_roi(self, frame):
        """Detect a region of interest - finger or face."""
        height, width = frame.shape[:2]
        roi = None
        
        # Try face detection first
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Use the largest face
                max_area = 0
                max_face = None
                for (x, y, w, h) in faces:
                    if w * h > max_area:
                        max_area = w * h
                        max_face = (x, y, w, h)
                        
                x, y, w, h = max_face
                
                # Define forehead region
                forehead_y = int(y + h * 0.1)
                forehead_h = int(h * 0.2)
                forehead_x = int(x + w * 0.25)
                forehead_w = int(w * 0.5)
                
                roi = {
                    'type': 'face',
                    'region': (forehead_x, forehead_y, forehead_w, forehead_h),
                    'display': (x, y, w, h)
                }
                return roi
        
        # Fallback: Define finger region
        finger_top = int(height * 0.3)
        finger_bottom = int(height * 0.7)
        finger_left = int(width * 0.3)
        finger_right = int(width * 0.7)
        
        roi = {
            'type': 'finger',
            'region': (finger_left, finger_top, finger_right - finger_left, finger_bottom - finger_top),
            'display': (finger_left, finger_top, finger_right - finger_left, finger_bottom - finger_top)
        }
        return roi
    
    def process_frame(self, frame):
        """Extract region of interest and process color channel values."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect ROI
        roi = self.detect_roi(frame)
        x, y, w, h = roi['region']
        
        # Extract region
        region = rgb_frame[y:y+h, x:x+w]
        
        if region.size == 0:
            return frame
        
        # Extract color channel values
        green_mean = np.mean(region[:, :, 1]) / (np.mean(region[:, :, 0]) + 1)
        red_mean = np.mean(region[:, :, 0])
        blue_mean = np.mean(region[:, :, 2])
        
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time() - self.start_time
        
        # Store values
        self.green_values.append(green_mean)
        self.red_values.append(red_mean)
        self.blue_values.append(blue_mean)
        self.times.append(current_time)
        
        # Maintain buffer size
        if len(self.green_values) > self.buffer_size:
            self.green_values.pop(0)
            self.red_values.pop(0)
            self.blue_values.pop(0)
            self.times.pop(0)
        
        # Draw ROI rectangle
        display_x, display_y, display_w, display_h = roi['display']
        
        if roi['type'] == 'face':
            cv2.rectangle(frame, (display_x, display_y), 
                         (display_x + display_w, display_y + display_h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Forehead ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Finger ROI", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                     
        return frame
    
    def compute_pulse(self):
        """Compute pulse rate from green channel signal."""
        if len(self.green_values) < self.buffer_size:
            return None, None
        
        if self.stable_pulse is not None:
            return None, self.stable_pulse
        
        signal = np.array(self.green_values)
        times_array = np.array(self.times)
        
        # Detrend signal
        polyfit = np.polyfit(times_array - times_array[0], signal, 2)
        signal = signal - np.polyval(polyfit, times_array - times_array[0])
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        # Bandpass filter
        b, a = self.butter_bandpass()
        filtered_signal = filtfilt(b, a, signal)
        
        # Find peaks
        peaks, _ = find_peaks(filtered_signal,
                               distance=int(self.fps * 0.5),
                               height=0.1,
                               prominence=0.2)
        
        if len(peaks) < 2:
            return filtered_signal, None
        
        peak_times = np.array([self.times[p] for p in peaks])
        intervals = np.diff(peak_times)
        valid_intervals = intervals[(intervals > 0.5) & (intervals < 1.5)]
        
        if len(valid_intervals) < 1:
            return filtered_signal, None
        
        mean_interval = np.mean(valid_intervals)
        current_pulse = int(round(60.0 / mean_interval))
        
        # Stability check
        self.pulse_history.append(current_pulse)
        if len(self.pulse_history) == self.pulse_history.maxlen:
            mean_pulse = np.mean(self.pulse_history)
            std_pulse = np.std(self.pulse_history)
            
            if std_pulse < 3.0:
                self.pulse_stable_count += 1
                if self.pulse_stable_count >= 5:
                    self.stable_pulse = int(round(mean_pulse))
                    return filtered_signal, self.stable_pulse
            else:
                self.pulse_stable_count = 0
        
        return filtered_signal, current_pulse
    
    def compute_spo2(self):
        """Compute SpO2 using red and blue channels."""
        if len(self.red_values) < self.buffer_size:
            return None, None, None
        
        if self.stable_spo2 is not None:
            return None, None, self.stable_spo2
        
        # Process red and blue signals
        red_signal = np.array(self.red_values)
        red_times = np.array(self.times)
        blue_signal = np.array(self.blue_values)
        
        # Detrend signals
        red_polyfit = np.polyfit(red_times - red_times[0], red_signal, 2)
        red_signal_detrended = red_signal - np.polyval(red_polyfit, red_times - red_times[0])
        
        blue_polyfit = np.polyfit(red_times - red_times[0], blue_signal, 2)
        blue_signal_detrended = blue_signal - np.polyval(blue_polyfit, red_times - red_times[0])
        
        # Normalize and filter signals
        red_signal_norm = (red_signal_detrended - np.mean(red_signal_detrended)) / np.std(red_signal_detrended)
        blue_signal_norm = (blue_signal_detrended - np.mean(blue_signal_detrended)) / np.std(blue_signal_detrended)
        
        b, a = self.butter_bandpass()
        filtered_red = filtfilt(b, a, red_signal_norm)
        filtered_blue = filtfilt(b, a, blue_signal_norm)
        
        # Signal quality assessment
        quality = self._assess_signal_quality(filtered_red, filtered_blue)
        
        if quality < 20:
            return filtered_red, filtered_blue, None
        
        # Find peaks
        red_peaks, _ = find_peaks(filtered_red, distance=int(self.fps * 0.5), height=0.1, prominence=0.2)
        blue_peaks, _ = find_peaks(filtered_blue, distance=int(self.fps * 0.5), height=0.1, prominence=0.2)
        
        if len(red_peaks) < 2 or len(blue_peaks) < 2:
            return filtered_red, filtered_blue, None
        
        # Calculate AC and DC components
        red_ac = np.max(filtered_red) - np.min(filtered_red)
        blue_ac = np.max(filtered_blue) - np.min(filtered_blue)
        red_dc = np.mean(self.red_values)
        blue_dc = np.mean(self.blue_values)
        
        # Store and average signal components
        self.red_ac_values.append(red_ac)
        self.blue_ac_values.append(blue_ac)
        self.red_dc_values.append(red_dc)
        self.blue_dc_values.append(blue_dc)
        
        avg_red_ac = np.mean(self.red_ac_values)
        avg_blue_ac = np.mean(self.blue_ac_values)
        avg_red_dc = np.mean(self.red_dc_values)
        avg_blue_dc = np.mean(self.blue_dc_values)
        
        # Calculate R and SpO2
        if avg_red_dc == 0 or avg_blue_dc == 0 or avg_red_ac == 0 or avg_blue_ac == 0:
            return filtered_red, filtered_blue, None
        
        r = (avg_red_ac / avg_red_dc) / (avg_blue_ac / avg_blue_dc)
        spo2 = 110 - 25 * r
        spo2 = max(min(spo2, 100), 80)
        
        current_spo2 = int(round(spo2))
        
        # Stability check
        self.spo2_history.append(current_spo2)
        if len(self.spo2_history) == self.spo2_history.maxlen:
            mean_spo2 = np.mean(self.spo2_history)
            std_spo2 = np.std(self.spo2_history)
            
            if std_spo2 < 1.5:
                self.spo2_stable_count += 1
                if self.spo2_stable_count >= 5:
                    self.stable_spo2 = int(round(mean_spo2))
                    return filtered_red, filtered_blue, self.stable_spo2
            else:
                self.spo2_stable_count = 0
        
        return filtered_red, filtered_blue, current_spo2
    
    def _assess_signal_quality(self, red_signal, blue_signal):
        """Assess signal quality based on multiple factors."""
        if len(red_signal) < 10 or len(blue_signal) < 10:
            return 0
        
        red_std = np.std(red_signal)
        blue_std = np.std(blue_signal)
        
        red_diffs = np.abs(np.diff(red_signal))
        blue_diffs = np.abs(np.diff(blue_signal))
        
        variance_score = min(50, max(0, (red_std + blue_std) * 100)) if red_std > 0.001 and blue_std > 0.001 else 0
        stability_score = max(0, 50 - np.mean(red_diffs + blue_diffs) * 100)
        
        quality = min(100, variance_score + stability_score)
        
        self.signal_quality = 0
