import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_angle_comparison(data, joint_name):
    plt.figure(figsize=(12, 6))
    plt.plot(data['timestamps'], data[f'left_{joint_name}'], label=f'Left {joint_name.title()}')
    plt.plot(data['timestamps'], data[f'right_{joint_name}'], label=f'Right {joint_name.title()}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angle (degrees)')
    plt.title(f'{joint_name.title()} Angle Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gait_cycle(data):
    right_knee = np.array(data['right_knee'])
    threshold = (np.max(right_knee) + np.min(right_knee)) / 2
    steps = np.where(right_knee > threshold)[0]

    if len(steps) >= 2:
        start, end = steps[0], steps[1]
        cycle_time = data['timestamps'][end] - data['timestamps'][start]
        norm_time = [(t - data['timestamps'][start]) / cycle_time * 100 
                     for t in data['timestamps'][start:end]]
        
        plt.figure(figsize=(14, 8))
        plt.plot(norm_time, data['left_knee'][start:end], label='Left Knee')
        plt.plot(norm_time, data['right_knee'][start:end], label='Right Knee')
        plt.plot(norm_time, data['left_ankle'][start:end], label='Left Ankle')
        plt.plot(norm_time, data['right_ankle'][start:end], label='Right Ankle')
        plt.xlabel('Gait Cycle Percentage')
        plt.ylabel('Angle (degrees)')
        plt.title('Joint Angles Through One Gait Cycle')
        plt.legend()
        plt.grid(True)
        plt.show()

def create_angle_animation(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.close()

    def update(frame):
        ax.clear()
        ax.set_xlim(0, data['timestamps'][-1])
        ax.set_ylim(0, 180)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Limb Angles Over Time')
        ax.plot(data['timestamps'][:frame], data['left_knee'][:frame], label='Left Knee')
        ax.plot(data['timestamps'][:frame], data['right_knee'][:frame], label='Right Knee')
        ax.plot(data['timestamps'][:frame], data['left_ankle'][:frame], label='Left Ankle')
        ax.plot(data['timestamps'][:frame], data['right_ankle'][:frame], label='Right Ankle')
        ax.legend()
        ax.grid(True)

    return FuncAnimation(fig, update, frames=len(data['timestamps']), interval=50)
