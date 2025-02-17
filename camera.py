pip install opencv-python numpy scipy matplotlib

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

class PulseDetector:
    def __init__(self, buffer_size=150, fps=30):
        self.buffer_size = buffer_size
        self.fps = fps
        self.green_values = []
        self.times = []
        self.start_time = None
        
    def butter_bandpass(self, lowcut=0.7, highcut=4.0, order=2):
        nyquist = self.fps / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def process_frame(self, frame):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Define forehead ROI (adjust these values based on your needs)
        height, width = frame.shape[:2]
        forehead_top = int(height * 0.1)
        forehead_bottom = int(height * 0.3)
        forehead_left = int(width * 0.3)
        forehead_right = int(width * 0.7)
        
        # Extract forehead region
        forehead_region = rgb_frame[forehead_top:forehead_bottom, 
                                  forehead_left:forehead_right]
        
        # Calculate mean green value
        green_mean = np.mean(forehead_region[:, :, 1])
        
        if self.start_time is None:
            self.start_time = cv2.getTickCount() / cv2.getTickFrequency()
            
        current_time = (cv2.getTickCount() / cv2.getTickFrequency()) - self.start_time
        
        # Update buffers
        self.green_values.append(green_mean)
        self.times.append(current_time)
        
        # Keep only recent values
        if len(self.green_values) > self.buffer_size:
            self.green_values.pop(0)
            self.times.pop(0)
        
        # Draw ROI on frame
        cv2.rectangle(frame, (forehead_left, forehead_top), 
                     (forehead_right, forehead_bottom), (0, 255, 0), 2)
        
        return frame
    
    def compute_pulse(self):
        if len(self.green_values) < self.buffer_size:
            return None, None
            
        # Detrend and normalize the signal
        signal = np.array(self.green_values)
        signal = signal - np.mean(signal)
        signal = signal / np.std(signal)
        
        # Apply bandpass filter
        b, a = self.butter_bandpass()
        filtered_signal = filtfilt(b, a, signal)
        
        # Find peaks
        peaks, _ = find_peaks(filtered_signal, distance=self.fps//2)
        
        if len(peaks) < 2:
            return None, None
            
        # Calculate pulse rate
        peak_times = np.array([self.times[p] for p in peaks])
        intervals = np.diff(peak_times)
        mean_interval = np.mean(intervals)
        pulse_rate = 60.0 / mean_interval
        
        return filtered_signal, int(round(pulse_rate))

def main():
    cap = cv2.VideoCapture(0)
    detector = PulseDetector()
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = detector.process_frame(frame)
        
        # Compute pulse rate
        filtered_signal, pulse_rate = detector.compute_pulse()
        
        # Display pulse rate on frame
        if pulse_rate is not None:
            cv2.putText(processed_frame, f"Pulse: {pulse_rate} BPM", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0), 2)
        
        # Update plots
        if filtered_signal is not None:
            ax1.clear()
            ax1.plot(detector.times, detector.green_values)
            ax1.set_title('Raw Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Green Channel Value')
            
            ax2.clear()
            ax2.plot(detector.times, filtered_signal)
            ax2.set_title('Filtered Signal')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Normalized Value')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        
        # Display frame
        cv2.imshow('Pulse Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    main()
