pip install opencv-python numpy scipy matplotlib

import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from collections import deque
class PulseDetector:
def __init__(self, buffer_size=300, fps=30):
self.buffer_size = buffer_size
self.fps = fps
self.green_values = []
self.times = []
self.start_time = None
self.stable_pulse = None
self.pulse_history = deque(maxlen=10) # Store last 10 readings
self.confidence_threshold = 0.8 # 80% confidence threshold
self.stable_count = 0
self.required_stable_readings = 5 # Number of consistent readings required
def butter_bandpass(self, lowcut=0.8, highcut=3.0, order=3):
nyquist = self.fps / 2
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(order, [low, high], btype='band')
return b, a
def is_reading_stable(self, current_pulse):
if current_pulse is None:
return False
self.pulse_history.append(current_pulse)
if len(self.pulse_history) < self.pulse_history.maxlen:
return False
# Calculate mean and standard deviation of recent readings
mean_pulse = np.mean(self.pulse_history)
std_pulse = np.std(self.pulse_history)
# Check if readings are consistent
if std_pulse < 3.0: # Standard deviation threshold
self.stable_count += 1

else:
self.stable_count = 0
# Return True if we have enough consistent readings
return self.stable_count >= self.required_stable_readings
def process_frame(self, frame):
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
height, width = frame.shape[:2]
forehead_top = int(height * 0.15)
forehead_bottom = int(height * 0.25)
forehead_left = int(width * 0.35)
forehead_right = int(width * 0.65)
forehead_region = rgb_frame[forehead_top:forehead_bottom,

forehead_left:forehead_right]

green_mean = np.mean(forehead_region[:, :, 1]) / (np.mean(forehead_region[:, :, 0]) + 1)
if self.start_time is None:
self.start_time = cv2.getTickCount() / cv2.getTickFrequency()
current_time = (cv2.getTickCount() / cv2.getTickFrequency()) - self.start_time
self.green_values.append(green_mean)
self.times.append(current_time)
if len(self.green_values) > self.buffer_size:
self.green_values.pop(0)
self.times.pop(0)
cv2.rectangle(frame, (forehead_left, forehead_top),
(forehead_right, forehead_bottom), (0, 255, 0), 2)
return frame
def compute_pulse(self):
if len(self.green_values) < self.buffer_size:
return None, None
# If we already have a stable pulse, return it
if self.stable_pulse is not None:
return None, self.stable_pulse

signal = np.array(self.green_values)
times_array = np.array(self.times)
polyfit = np.polyfit(times_array - times_array[0], signal, 2)
signal = signal - np.polyval(polyfit, times_array - times_array[0])
signal = (signal - np.mean(signal)) / np.std(signal)
b, a = self.butter_bandpass()
filtered_signal = filtfilt(b, a, signal)
peaks, _ = find_peaks(filtered_signal,
distance=int(self.fps * 0.5),
height=0.1,
prominence=0.2)

if len(peaks) < 2:
return filtered_signal, None
peak_times = np.array([self.times[p] for p in peaks])
intervals = np.diff(peak_times)
valid_intervals = intervals[
(intervals > 0.5) &
(intervals < 1.5)
]
if len(valid_intervals) < 1:
return filtered_signal, None
mean_interval = np.mean(valid_intervals)
current_pulse = int(round(60.0 / mean_interval))
# Check if the reading is stable
if self.is_reading_stable(current_pulse):
self.stable_pulse = int(round(np.mean(self.pulse_history)))
return filtered_signal, self.stable_pulse
return filtered_signal, current_pulse
def main():
cap = cv2.VideoCapture(0)
detector = PulseDetector()

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
while True:
ret, frame = cap.read()
if not ret:
break
processed_frame = detector.process_frame(frame)
filtered_signal, pulse_rate = detector.compute_pulse()
# Display status on frame
if detector.stable_pulse is not None:
cv2.putText(processed_frame, f"Stable Pulse: {detector.stable_pulse} BPM",
(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
(0, 255, 0), 2)
elif pulse_rate is not None:
cv2.putText(processed_frame, f"Measuring... Current: {pulse_rate} BPM",
(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
(0, 255, 0), 2)
else:
cv2.putText(processed_frame, "Measuring...",
(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
(0, 255, 0), 2)
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
cv2.imshow('Pulse Detection', processed_frame)
if cv2.waitKey(1) & 0xFF == ord('q'):

break
cap.release()
cv2.destroyAllWindows()
plt.close()
if __name__ == "__main__":
main()
