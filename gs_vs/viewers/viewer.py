import matplotlib
#matplotlib.use("Agg")  # must come before importing pyplot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
from matplotlib.cm import get_cmap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tools.image_tools import image_difference
import os

plt.style.use('dark_background')


class Viewer:
    def __init__(self, start_image, target_image, start_pose, target_pose,
                 save_video=True, video_filename="./logs/visual_servoing.avi", fps=5):
        self.start_image = start_image
        self.target_image = target_image
        self.start_pose = start_pose
        self.target_pose = target_pose
        self.poses = [start_pose]
        self.errors = []
        self.control_velocities = []
        self.save_video = save_video
        self.video_filename = video_filename
        self.fps = fps
        self.video_writer = None

        # Initialize figure and canvas
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        # Create subplots
        self.ax_start_image = self.fig.add_subplot(2, 4, 1)
        self.ax_target_image = self.fig.add_subplot(2, 4, 2)
        self.ax_current_image = self.fig.add_subplot(2, 4, 3)
        self.ax_image_diff = self.fig.add_subplot(2, 4, 4)
        self.ax_control_velocity = self.fig.add_subplot(2, 4, 5)
        self.ax_pose_evolution = self.fig.add_subplot(2, 4, (7, 8), projection='3d')
        self.ax_error_evolution = self.fig.add_subplot(2, 4, 6)

        # 3D pose view
        self.ax_pose_evolution.set_title('Pose Evolution')
        self.ax_pose_evolution.set_xlabel('X')
        self.ax_pose_evolution.set_ylabel('Y')
        self.ax_pose_evolution.set_zlabel('Z')
        self.ax_pose_evolution.grid(True)

        # Plot start & target
        self.plot_pose(self.target_pose, self.ax_pose_evolution, label='Target Pose', color='yellow')
        self.plot_pose(self.start_pose, self.ax_pose_evolution, label='Start Pose', color='blue')
        self.ax_pose_evolution.legend()

        if matplotlib.get_backend().lower() != "agg":
            plt.pause(0.001)

    def plot_pose(self, pose, axis, label=None, color='blue', alpha=1.0, s=50):
        origin = pose[:3, 3]
        axis.scatter(*origin, color=color, label=label, alpha=alpha, s=s)

    def update(self, current_image, iteration, current_pose, total_error, control_velocity, mask=None):
        self.poses.append(current_pose)
        self.errors.append(total_error)
        self.control_velocities.append(control_velocity)

        # Update image panels
        self.ax_start_image.imshow(self.start_image, cmap='gray'); self.ax_start_image.set_title('Start Image'); self.ax_start_image.axis('off')
        self.ax_target_image.imshow(self.target_image, cmap='gray'); self.ax_target_image.set_title('Target Image'); self.ax_target_image.axis('off')
        self.ax_current_image.imshow(current_image, cmap='gray'); self.ax_current_image.set_title('Current Image'); self.ax_current_image.axis('off')

        image_diff = image_difference(current_image, self.target_image)
        self.ax_image_diff.imshow(image_diff, cmap='gray')
        self.ax_image_diff.set_title('Difference = Current - Target')
        self.ax_image_diff.axis('off')

        # Pose evolution
        self.ax_pose_evolution.cla()
        self.ax_pose_evolution.set_title('Pose Evolution')
        self.ax_pose_evolution.set_xlabel('X'); self.ax_pose_evolution.set_ylabel('Y'); self.ax_pose_evolution.set_zlabel('Z')
        self.ax_pose_evolution.grid(True)
        for pose in self.poses:
            self.plot_pose(pose, self.ax_pose_evolution, color='red', alpha=1.0, s=25)
        self.plot_pose(self.target_pose, self.ax_pose_evolution, label='Target', color='yellow')
        self.plot_pose(self.start_pose, self.ax_pose_evolution, label='Start', color='blue')
        self.plot_pose(current_pose, self.ax_pose_evolution, label='Current', color='red')
        self.ax_pose_evolution.legend()

        # Error evolution
        self.ax_error_evolution.cla()
        self.ax_error_evolution.plot(self.errors, color='red', label='Error Evolution')
        self.ax_error_evolution.legend()

        # Control velocities
        cmap = get_cmap('tab10')
        colors = [cmap(i) for i in range(6)]
        self.ax_control_velocity.cla()
        vels = np.array(self.control_velocities)
        if len(vels) > 0:
            labels_t = ['vx', 'vy', 'vz']
            labels_r = ['ωx', 'ωy', 'ωz']
            for i in range(3):
                self.ax_control_velocity.plot(vels[:, i], color=colors[i], label=labels_t[i])
            for i in range(3, 6):
                self.ax_control_velocity.plot(vels[:, i], color=colors[i], label=labels_r[i-3], linestyle='dashed')
            self.ax_control_velocity.legend()

        self.fig.suptitle(f'Iteration {iteration} | Error: {total_error:.2f}')

        # Write frame
        if self.save_video:
            self.save_frame()
        elif matplotlib.get_backend().lower() != "agg":
            plt.pause(0.00001)

    def save_frame(self):
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.canvas.get_width_height()
        frame = buf.reshape((h, w, 3))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.video_writer is None:
            os.makedirs(os.path.dirname(self.video_filename), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_filename = self.video_filename.replace('.avi', '.mp4')
            self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, self.fps, (w, h))
            if not self.video_writer.isOpened():
                raise RuntimeError(f"[ERROR] Could not open VideoWriter for {self.video_filename}")
            print(f"[INFO] VideoWriter initialized: {self.video_filename} ({w}x{h})")

        self.video_writer.write(frame_bgr)

    def close(self):
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            print(f"[INFO] Video saved as {self.video_filename}")
        if matplotlib.get_backend().lower() != "agg":
            plt.show()

