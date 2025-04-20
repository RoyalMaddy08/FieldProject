from video_processing.process_video import process_video
from visualization.plots import plot_angle_comparison, plot_gait_cycle, create_angle_animation

# Process your video
video_path = 'walking_video.mp4'  # Replace with your video path
angle_data = process_video(video_path)

# Generate visualizations
plot_angle_comparison(angle_data, 'knee')
plot_angle_comparison(angle_data, 'ankle')
plot_angle_comparison(angle_data, 'hip')
plot_gait_cycle(angle_data)

# Optional: Save animation
# ani = create_angle_animation(angle_data)
# ani.save('gait_analysis.mp4', writer='ffmpeg', fps=30)
