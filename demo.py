pip install opencv-python numpy scipy matplotlib


import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque
import time
import os

class SpO2Detector:
    def __init__(self, buffer_size=300, fps=30):
        """Initialize the SpO2 detector with buffer size and FPS."""
        self.buffer_size = buffer_size
        self.fps = fps
        
        # Store red and blue channel values
        self.red_values = []
        self.blue_values = []
        self.times = []
        self.start_time = None
        
        # SpO2 calculation values
        self.spo2_readings = []
        self.stable_spo2 = None
        self.spo2_history = deque(maxlen=10)  # Store last 10 readings
        self.stable_count = 0
        self.required_stable_readings = 5  # Number of consistent readings required
        
        # Signal quality assessment
        self.signal_quality = 0  # 0-100 scale
        
        # Calculate AC and DC components
        self.red_ac_values = deque(maxlen=10)
        self.blue_ac_values = deque(maxlen=10)
        self.red_dc_values = deque(maxlen=10)
        self.blue_dc_values = deque(maxlen=10)
        
        # Try to load face cascade - handle errors gracefully
        self.face_cascade = None
        try:
            # Try multiple potential paths for the Haar cascade file
            cascade_paths = [
                'haarcascade_frontalface_default.xml',  # Local directory
                os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml'),  # OpenCV installation
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',  # Linux common path
                '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',  # Another Linux path
                'C:/opencv/haarcascades/haarcascade_frontalface_default.xml'  # Windows potential path
            ]
            
            for path in cascade_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    break
                    
            # If still not loaded, download the file
            if self.face_cascade is None or self.face_cascade.empty():
                print("Face cascade not found. Face detection disabled.")
                self.face_cascade = None
        except Exception as e:
            print(f"Could not load face cascade: {e}")
            self.face_cascade = None
            
    def butter_bandpass(self, lowcut=0.8, highcut=3.0, order=3):
        """Create a bandpass filter for heart rate signal processing."""
        nyquist = self.fps / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
        
    def is_reading_stable(self, current_spo2):
        """Check if the SpO2 reading is stable based on recent values."""
        if current_spo2 is None:
            return False
            
        self.spo2_history.append(current_spo2)
        if len(self.spo2_history) < self.spo2_history.maxlen:
            return False
            
        # Calculate mean and standard deviation of recent readings
        mean_spo2 = np.mean(self.spo2_history)
        std_spo2 = np.std(self.spo2_history)
        
        # Check if readings are consistent
        if std_spo2 < 1.5:  # Standard deviation threshold for SpO2
            self.stable_count += 1
        else:
            self.stable_count = 0
            
        # Return True if we have enough consistent readings
        return self.stable_count >= self.required_stable_readings
        
    def assess_signal_quality(self, red_signal, blue_signal):
        """Assess signal quality based on multiple factors"""
        if len(red_signal) < 10 or len(blue_signal) < 10:
            return 0
            
        # Check signal variance (too low = poor signal)
        red_std = np.std(red_signal)
        blue_std = np.std(blue_signal)
        
        # Check for rapid large changes (possible motion artifacts)
        red_diffs = np.abs(np.diff(red_signal))
        blue_diffs = np.abs(np.diff(blue_signal))
        
        # Higher quality when we have moderate variance but not extreme changes
        variance_score = min(50, max(0, (red_std + blue_std) * 100)) if red_std > 0.001 and blue_std > 0.001 else 0
        stability_score = max(0, 50 - np.mean(red_diffs + blue_diffs) * 100)
        
        quality = min(100, variance_score + stability_score)
        
        # Exponential moving average for smoothing quality indicator
        self.signal_quality = 0.7 * self.signal_quality + 0.3 * quality
        return int(self.signal_quality)
    
    def detect_roi(self, frame):
        """Detect a region of interest - finger or face."""
        height, width = frame.shape[:2]
        roi = None
        
        # Try face detection first if cascade is available
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
                
                # Define forehead region from face
                forehead_y = int(y + h * 0.1)
                forehead_h = int(h * 0.2)
                forehead_x = int(x + w * 0.25)
                forehead_w = int(w * 0.5)
                
                roi = {
                    'type': 'face',
                    'region': (forehead_x, forehead_y, forehead_w, forehead_h),
                    'display': (x, y, w, h)  # For rectangle display
                }
                return roi
        
        # Fallback: Define finger region (center of frame)
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
        """Extract region of interest and process the red and blue channel values."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect ROI (face or finger)
        roi = self.detect_roi(frame)
        x, y, w, h = roi['region']
        
        # Extract region
        region = rgb_frame[y:y+h, x:x+w]
        
        # Skip processing if region is empty
        if region.size == 0:
            return frame
        
        # Extract red and blue channel values
        red_mean = np.mean(region[:, :, 0])
        blue_mean = np.mean(region[:, :, 2])
        
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time() - self.start_time
        
        self.red_values.append(red_mean)
        self.blue_values.append(blue_mean)
        self.times.append(current_time)
        
        if len(self.red_values) > self.buffer_size:
            self.red_values.pop(0)
            self.blue_values.pop(0)
            self.times.pop(0)
            
        # Draw ROI rectangle with appropriate label
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
        
    def compute_spo2(self):
        """Process the signals to compute SpO2."""
        if len(self.red_values) < self.buffer_size:
            return None, None, None
            
        # If we already have a stable SpO2, return it
        if self.stable_spo2 is not None:
            return None, None, self.stable_spo2
            
        # Process red signal
        red_signal = np.array(self.red_values)
        red_times = np.array(self.times)
        
        # Process blue signal
        blue_signal = np.array(self.blue_values)
        
        # Detrend signals (remove slow trends)
        red_polyfit = np.polyfit(red_times - red_times[0], red_signal, 2)
        red_signal_detrended = red_signal - np.polyval(red_polyfit, red_times - red_times[0])
        
        blue_polyfit = np.polyfit(red_times - red_times[0], blue_signal, 2)
        blue_signal_detrended = blue_signal - np.polyval(blue_polyfit, red_times - red_times[0])
        
        # Normalize signals
        red_signal_norm = (red_signal_detrended - np.mean(red_signal_detrended)) / (np.std(red_signal_detrended) if np.std(red_signal_detrended) > 0 else 1)
        blue_signal_norm = (blue_signal_detrended - np.mean(blue_signal_detrended)) / (np.std(blue_signal_detrended) if np.std(blue_signal_detrended) > 0 else 1)
        
        # Apply bandpass filter
        b, a = self.butter_bandpass()
        filtered_red = filtfilt(b, a, red_signal_norm)
        filtered_blue = filtfilt(b, a, blue_signal_norm)
        
        # Assess quality before computing SpO2
        quality = self.assess_signal_quality(filtered_red, filtered_blue)
        
        # Only proceed if quality is reasonable
        if quality < 20:
            return filtered_red, filtered_blue, None
        
        # Find peaks for both signals
        red_peaks, _ = find_peaks(filtered_red, distance=int(self.fps * 0.5), height=0.1, prominence=0.2)
        blue_peaks, _ = find_peaks(filtered_blue, distance=int(self.fps * 0.5), height=0.1, prominence=0.2)
        
        if len(red_peaks) < 2 or len(blue_peaks) < 2:
            return filtered_red, filtered_blue, None
            
        # Calculate AC and DC components for both signals
        red_ac = np.max(filtered_red) - np.min(filtered_red)
        blue_ac = np.max(filtered_blue) - np.min(filtered_blue)
        
        red_dc = np.mean(self.red_values)
        blue_dc = np.mean(self.blue_values)
        
        # Store AC and DC values for averaging
        self.red_ac_values.append(red_ac)
        self.blue_ac_values.append(blue_ac)
        self.red_dc_values.append(red_dc)
        self.blue_dc_values.append(blue_dc)
        
        # Calculate average AC and DC values for more stable readings
        avg_red_ac = np.mean(self.red_ac_values)
        avg_blue_ac = np.mean(self.blue_ac_values)
        avg_red_dc = np.mean(self.red_dc_values)
        avg_blue_dc = np.mean(self.blue_dc_values)
        
        # Calculate R (ratio of ratios)
        if avg_red_dc == 0 or avg_blue_dc == 0 or avg_red_ac == 0 or avg_blue_ac == 0:
            return filtered_red, filtered_blue, None
            
        r = (avg_red_ac / avg_red_dc) / (avg_blue_ac / avg_blue_dc)
        
        # Convert R to SpO2 using empirical formula
        # Note: This is an approximation. Actual SpO2 devices use calibrated lookup tables
        # SpO2 = 110 - 25 * R is a simplified approximation
        spo2 = 110 - 25 * r
        
        # Clamp to realistic SpO2 range
        spo2 = max(min(spo2, 100), 80)
        
        current_spo2 = int(round(spo2))
        
        # Check if the reading is stable
        if self.is_reading_stable(current_spo2):
            self.stable_spo2 = int(round(np.mean(self.spo2_history)))
            return filtered_red, filtered_blue, self.stable_spo2
            
        return filtered_red, filtered_blue, current_spo2

def main():
    """Main function to capture video and detect SpO2 in real time."""
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        # Estimate actual FPS
        num_frames = 30
        start = time.time()
        for i in range(num_frames):
            ret, _ = cap.read()
            if not ret:
                break
        end = time.time()
        actual_fps = num_frames / (end - start) if (end - start) > 0 else 30
        
        # Reset camera to ensure we're at the beginning
        cap.release()
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
            
        # Validate FPS
        if actual_fps <= 0 or actual_fps > 100:  # Invalid FPS
            actual_fps = 30  # Fallback
            
        print(f"Detected camera FPS: {actual_fps:.1f}")
        
        # Initialize detector with actual FPS
        detector = SpO2Detector(fps=actual_fps)
        
        # Enable interactive plotting
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            processed_frame = detector.process_frame(frame)
            filtered_red, filtered_blue, spo2 = detector.compute_spo2()
            
            # Display SpO2 status on frame
            if detector.stable_spo2 is not None:
                cv2.putText(processed_frame, f"Stable SpO2: {detector.stable_spo2}%", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                quality_text = f"Signal Quality: {detector.signal_quality}%"
                cv2.putText(processed_frame, quality_text, 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif spo2 is not None:
                cv2.putText(processed_frame, f"Measuring... Current: {spo2}%", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                quality_text = f"Signal Quality: {detector.signal_quality}%"
                cv2.putText(processed_frame, quality_text, 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "Measuring signal...", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                quality_text = f"Signal Quality: {detector.signal_quality}%"
                cv2.putText(processed_frame, quality_text, 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display mode instruction
            if detector.face_cascade is not None:
                cv2.putText(processed_frame, "Mode: Face or finger detection", 
                           (20, processed_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "Mode: Finger detection only", 
                           (20, processed_frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            # Show instructions
            cv2.putText(processed_frame, "Place finger over camera or position your face. Keep still.", 
                       (20, processed_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Plot signals if available
            if filtered_red is not None and filtered_blue is not None and len(detector.times) > 0:
                # Plot only the last 150 samples for clearer visualization
                plot_length = min(150, len(detector.times))
                times_to_plot = detector.times[-plot_length:]
                red_to_plot = filtered_red[-plot_length:]
                blue_to_plot = filtered_blue[-plot_length:]
                
                # Adjust time to start from 0
                if len(times_to_plot) > 0:
                    times_to_plot = np.array(times_to_plot) - times_to_plot[0]
                
                # Plot red signal
                ax1.clear()
                ax1.plot(times_to_plot, red_to_plot, 'r-', label="Red Channel")
                ax1.set_title('Red Channel Signal')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Normalized Value')
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right')
                
                # Plot blue signal with pulse indicators
                ax2.clear()
                ax2.plot(times_to_plot, blue_to_plot, 'b-', label="Blue Channel", linewidth=2)
                
                # Mark detected peaks if we have them in the visible range
                if spo2 is not None:
                    # Find peaks in the visible range
                    visible_peaks, _ = find_peaks(
                        blue_to_plot,
                        distance=int(detector.fps * 0.5),
                        height=0.1,
                        prominence=0.2
                    )
                    
                    if len(visible_peaks) > 0:
                        peak_times = [times_to_plot[p] for p in visible_peaks]
                        peak_values = [blue_to_plot[p] for p in visible_peaks]
                        ax2.plot(peak_times, peak_values, 'go', markersize=8, label="Pulses")
                
                ax2.set_title(f'Blue Channel Signal - {"Stable" if detector.stable_spo2 else "Measuring"}')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylabel('Amplitude')
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')
                
                # Update plot layout
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)
            
            # Display the processed frame
            cv2.imshow('SpO2 Detection', processed_frame)
            
            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

if __name__ == "__main__":
    main()
