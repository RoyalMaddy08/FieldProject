!pip install opencv-python numpy scipy

import cv2
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import time
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Constants
FACE_DETECTION_MODEL = "haarcascade_frontalface_default.xml"
FOREHEAD_RATIO = 0.25  # Ratio of forehead height to face height
FPS = 30  # Webcam frame rate
BUFFER_SIZE = 150  # Number of frames to analyze for BPM calculation
MIN_BPM = 40  # Minimum valid BPM
MAX_BPM = 180  # Maximum valid BPM

# Bandpass filter for heart rate signal
def bandpass_filter(signal, lowcut=0.8, highcut=3.0, fs=FPS, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_DETECTION_MODEL)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Webcam not found!")

# Buffer for storing green channel intensities
green_buffer = []
bpm = 0
last_face_time = time.time()

# Create a matplotlib figure for displaying the video feed
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Pulse Detector")
ax.axis('off')
img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

            # Extract forehead region
            forehead_h = int(h * FOREHEAD_RATIO)
            forehead = frame[y:y + forehead_h, x:x + w]

            # Extract green channel intensity
            green_intensity = np.mean(forehead[:, :, 1])  # Green channel is index 1
            green_buffer.append(green_intensity)

            # Keep buffer size fixed
            if len(green_buffer) > BUFFER_SIZE:
                green_buffer.pop(0)

            # Draw forehead region
            cv2.rectangle(frame, (x, y), (x + w, y + forehead_h), (0, 255, 0), 2)

            # Calculate BPM if enough data is available
            if len(green_buffer) == BUFFER_SIZE:
                # Apply bandpass filter
                filtered_signal = bandpass_filter(green_buffer)

                # Find peaks in the filtered signal
                peaks, _ = find_peaks(filtered_signal, distance=FPS * 2)
                if len(peaks) >= 2:
                    peak_times = np.array(peaks) / FPS
                    bpm = 60 / np.mean(np.diff(peak_times))
                    bpm = np.clip(bpm, MIN_BPM, MAX_BPM)  # Clip to valid range

            last_face_time = time.time()
        else:
            # Reset BPM if no face is detected
            if time.time() - last_face_time > 1:  # 1-second delay before reset
                bpm = 0
                green_buffer = []

        # Display BPM on the frame
        cv2.putText(frame, f"BPM: {int(bpm)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the display in Jupyter Notebook
        img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        clear_output(wait=True)
        display(fig)
        plt.pause(0.01)  # Small delay to allow the display to update

except KeyboardInterrupt:
    print("Stopping the pulse detector...")

finally:
    # Release resources
    cap.release()
    plt.close()
