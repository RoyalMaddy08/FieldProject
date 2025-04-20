import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from utils.angle_math import calculate_angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamps = []
    left_knee_angles, right_knee_angles = [], []
    left_ankle_angles, right_ankle_angles = [], []
    left_hip_angles, right_hip_angles = [], []
    
    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        timestamp = frame_count / fps
        frame_count += 1

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            get = lambda l: [landmarks[l.value].x, landmarks[l.value].y]
            left_hip, right_hip = get(mp_pose.PoseLandmark.LEFT_HIP), get(mp_pose.PoseLandmark.RIGHT_HIP)
            left_knee, right_knee = get(mp_pose.PoseLandmark.LEFT_KNEE), get(mp_pose.PoseLandmark.RIGHT_KNEE)
            left_ankle, right_ankle = get(mp_pose.PoseLandmark.LEFT_ANKLE), get(mp_pose.PoseLandmark.RIGHT_ANKLE)

            left_knee_angles.append(calculate_angle(left_hip, left_knee, left_ankle))
            right_knee_angles.append(calculate_angle(right_hip, right_knee, right_ankle))
            left_ankle_angles.append(calculate_angle(left_knee, left_ankle, [left_ankle[0], left_ankle[1]-0.1]))
            right_ankle_angles.append(calculate_angle(right_knee, right_ankle, [right_ankle[0], right_ankle[1]-0.1]))
            left_hip_angles.append(calculate_angle([left_hip[0], left_hip[1]-0.1], left_hip, left_knee))
            right_hip_angles.append(calculate_angle([right_hip[0], right_hip[1]-0.1], right_hip, right_knee))
            timestamps.append(timestamp)

    cap.release()

    window_size = min(15, len(left_knee_angles)//2)
    if window_size > 2 and window_size % 2 == 1:
        apply_filter = lambda x: savgol_filter(x, window_size, 2)
        left_knee_angles, right_knee_angles = apply_filter(left_knee_angles), apply_filter(right_knee_angles)
        left_ankle_angles, right_ankle_angles = apply_filter(left_ankle_angles), apply_filter(right_ankle_angles)
        left_hip_angles, right_hip_angles = apply_filter(left_hip_angles), apply_filter(right_hip_angles)

    return {
        'timestamps': timestamps,
        'left_knee': left_knee_angles,
        'right_knee': right_knee_angles,
        'left_ankle': left_ankle_angles,
        'right_ankle': right_ankle_angles,
        'left_hip': left_hip_angles,
        'right_hip': right_hip_angles
    }
