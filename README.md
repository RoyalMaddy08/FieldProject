# Field Project
This project performs gait analysis by extracting and analyzing lower limb joint angles from a video of a walking subject. It uses MediaPipe's Pose Estimation, OpenCV, and Matplotlib to track and visualize the motion of the hips, knees, and ankles throughout the gait cycle.
Developed a gait analysis tool leveraging MediaPipe’s pose estimation and signal processing techniques to calculate and visualize joint angles of the lower body during walking. The system analyzes knee, hip, and ankle dynamics, identifies gait cycles, and provides comparative visualizations of left vs. right limb behavior. Additionally, angle smoothing using Savitzky-Golay filter and animation of joint motion were implemented to enhance interpretability.

Demo
The system takes a walking video as input and outputs:

Angle plots of left and right knee, hip, and ankle

A gait cycle graph showing angle changes across a step

An animated visualization of limb angles over time


Libraries Used
OpenCV for video processing

MediaPipe for pose landmark detection

NumPy for vector math and angle calculations

SciPy for smoothing and peak detection

Matplotlib for plotting and animations


How It Works
Pose Estimation: MediaPipe detects body landmarks for each frame.

Joint Angle Calculation: Angles at hip, knee, and ankle are calculated using 2D vector math.

Data Smoothing: Optional Savitzky-Golay filter is applied to reduce noise.

Gait Cycle Detection: Knee angles are used to segment a gait cycle.

Visualization: Time-series plots, gait cycle plots, and animated graphs are generated.


Outputs
✅ Angle Comparison
Plots of left vs right angles for:

Knee

Ankle

Hip

✅ Gait Cycle Plot
A normalized plot showing joint angles across one gait cycle.

✅ Animated Graph (Optional)
Live animation showing how angles change over time.

Notes
Ensure the video has a clear side view of walking for accurate joint detection.

Real-time performance depends on video quality and pose detection confidence.

Adjust smoothing window_size if needed for better accuracy.

Applications
Medical gait analysis

Sports performance tracking

Rehabilitation monitoring

Biomechanics education

