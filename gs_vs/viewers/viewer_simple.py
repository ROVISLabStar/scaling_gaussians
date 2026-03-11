import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
import os
from tools.image_tools import image_difference


class Viewer:
    """
    2x3 Visual Servoing Viewer:
    ---------------------------
    [Start]     [Target]     [Current]
    [Diff]      [Velocity]   [Error]

    Saves each frame into a video.
    """

    def __init__(self, start_image, target_image,
                 video_filename="./logs/visual_servoing.mp4",
                 fps=5):

        self.start_image = start_image
        self.target_image = target_image
        self.video_filename = video_filename
        self.fps = fps
        self.video_writer = None

        # Logs
        self.velocities = []
        self.errors = []

        os.makedirs(os.path.dirname(video_filename), exist_ok=True)

        # Create 2x3 figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.canvas = FigureCanvas(self.fig)

        titles = [
            "Initial Real Image", "Desired Virtual Image", "Current Real Image",
            "Difference = Current - Desired", "Velocity Evolution", "Photometric Error"
        ]

        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title)
            ax.axis("off")

        if matplotlib.get_backend().lower() != "agg":
            plt.pause(0.001)

    def update(self, current_image, iteration, error=None, velocity=None):

        # Store history
        if error is not None:
            self.errors.append(error)
        if velocity is not None:
            self.velocities.append(velocity)

        # Compute difference image
        diff = image_difference(current_image, self.target_image)

        # ------------------------
        # Update image panels
        # ------------------------
        self.axes[0, 0].imshow(self.start_image, cmap="gray")
        self.axes[0, 1].imshow(self.target_image, cmap="gray")
        self.axes[0, 2].imshow(current_image, cmap="gray")
        self.axes[1, 0].imshow(diff, cmap="gray")

        # ------------------------
        # Velocity evolution plot
        # ------------------------
        ax_vel = self.axes[1, 1]
        ax_vel.cla()
        ax_vel.set_title("Velocity Evolution")
        ax_vel.set_xlabel("Iteration")
        ax_vel.set_ylabel("Velocities")
        ax_vel.grid(True)

        if len(self.velocities) > 1:
            vels = np.array(self.velocities)
            labels = ["vx", "vy", "vz", "ωx", "ωy", "ωz"]

            for i in range(6):
                ax_vel.plot(vels[:, i], label=labels[i])
            ax_vel.legend()

        # ------------------------
        # Error evolution plot
        # ------------------------
        ax_err = self.axes[1, 2]
        ax_err.cla()
        ax_err.set_title("Photometric Error")
        ax_err.set_xlabel("Iteration")
        ax_err.set_ylabel("||e||²")
        ax_err.grid(True)

        if len(self.errors) > 1:
            ax_err.plot(self.errors, color='red')

        # ------------------------
        # Write video frame
        # ------------------------
        self.fig.suptitle(f"Iteration {iteration}")

        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.canvas.get_width_height()
        frame = buf.reshape((h, w, 3))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.video_filename, fourcc, self.fps, (w, h)
            )

            if not self.video_writer.isOpened():
                raise RuntimeError(f"[ERROR] Could not open VideoWriter for {self.video_filename}")

            print(f"[INFO] VideoWriter initialized: {self.video_filename}")

        self.video_writer.write(frame_bgr)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[INFO] Video saved to {self.video_filename}")

        if matplotlib.get_backend().lower() != "agg":
            plt.show()

