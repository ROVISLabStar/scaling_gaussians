import numpy as np
import viser
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from viser import transforms as tf

class VsViserViewer:
    """
    Viser-based viewer for visual servoing with gsplat.
    Uses camera frustums to show camera poses with images.
    """

    def __init__(
        self,
        rgb_des,
        cdMo,
        rgb_cur,
        cMo,
        image_size,
        aspect_ratio,
        server_port=8080,
        image_scale=1.0,
        means=None,
        scales=None,
        quats=None,
        sh0=None,
        opacities=None,
        downsample_factor=1,
        axes_length=0.1,
        axes_radius=0.005,
    ):
        """
        Args:
            rgb_des: (H,W,3) float32 in [0,1] - desired/target image
            cdMo: desired camera pose (4x4, world->camera)
            rgb_cur: (H,W,3) float32 in [0,1] - current image
            cMo: current camera pose (4x4, world->camera)
            image_size: (W, H)
            aspect_ratio: W / H
            downsample_factor: factor to downsample images in frustums
            axes_length: length of coordinate axes arrows
            axes_radius: radius of coordinate axes arrows
        """

        self.W, self.H = image_size
        self.aspect_ratio = aspect_ratio
        self.downsample_factor = downsample_factor
        self.axes_length = axes_length
        self.axes_radius = axes_radius

        # Ensure images are numpy arrays
        if hasattr(rgb_des, 'cpu'):
            rgb_des = rgb_des.cpu().numpy()
        if hasattr(rgb_cur, 'cpu'):
            rgb_cur = rgb_cur.cpu().numpy()
        
        rgb_des = np.clip(rgb_des, 0.0, 1.0)
        rgb_cur = np.clip(rgb_cur, 0.0, 1.0)

        # -------------------------
        # Viser server
        # -------------------------
        self.server = viser.ViserServer(port=server_port)
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large")
        print(f"[INFO] Open http://localhost:{server_port}")

        # Fix world up direction (COLMAP convention)
        up_world = cdMo[:3, :3] @ np.array([0.0, -1.0, 0.0])
        up_world /= np.linalg.norm(up_world)
        self.server.scene.set_up_direction(tuple(up_world))
        
        # Add world frame with axes
        """
        self.world_frame = self.server.scene.add_frame(
            "/world",
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        """
        # -------------------------
        # Camera frustums with axes
        # -------------------------
        
        # Compute FoV (using approximate formula)
        # For fisheye, typical FoV is 180° = π radians
        fov = np.pi/2  # 180 degrees in radians
        
        # Downsample images for frustums
        rgb_des_down = rgb_des[::downsample_factor, ::downsample_factor]
        rgb_cur_down = rgb_cur[::downsample_factor, ::downsample_factor]

        # Desired/target camera frustum (green)
        self.frustum_des = self.server.scene.add_camera_frustum(
            "/frustums/desired",
            fov=fov,
            aspect=aspect_ratio,
            scale=image_scale,
            image=rgb_des_down,
            wxyz=tf.SO3.from_matrix(cdMo[:3, :3]).wxyz,
            position=tuple(cdMo[:3, 3]),
            color=(0, 0, 0),  # Green
        )
 
        # Add axes to desired frustum
        self.axes_des = self.server.scene.add_frame(
            "/frustums/desired/axes",
            axes_length=axes_length,
            axes_radius=axes_radius,
        )

        # Initial camera frustum (red)
        self.frustum_init = self.server.scene.add_camera_frustum(
            "/frustums/initial",
            fov=fov,
            aspect=aspect_ratio,
            scale=image_scale,
            image=rgb_cur_down,
            wxyz=tf.SO3.from_matrix(cMo[:3, :3]).wxyz,
            position=tuple(cMo[:3, 3]),
            color=(0, 0, 0),  # Red
        )
        
        # Add axes to initial frustum
        self.axes_init = self.server.scene.add_frame(
            "/frustums/initial/axes",
            axes_length=axes_length,
            axes_radius=axes_radius,
        )

        # Current camera frustum (blue, will be updated)
        self.frustum_cur = self.server.scene.add_camera_frustum(
            "/frustums/current",
            fov=fov,
            aspect=aspect_ratio,
            scale=image_scale,
            image=rgb_cur_down,
            wxyz=tf.SO3.from_matrix(cMo[:3, :3]).wxyz,
            position=tuple(cMo[:3, 3]),
            color=(0, 0, 0),  # Blue
        )
        
        # Add axes to current frustum
        self.axes_cur = self.server.scene.add_frame(
            "/frustums/current/axes",
            axes_length=axes_length,
            axes_radius=axes_radius,
        )


        # -------------------------
        # GUI controls
        # -------------------------
        # Toggle visibility of frustums and axes
        self.toggle_des = self.server.gui.add_checkbox(
            label="Show Desired Image",
            initial_value=True
        )
        self.toggle_init = self.server.gui.add_checkbox(
            label="Show Initial Image",
            initial_value=True
        )
        self.toggle_cur = self.server.gui.add_checkbox(
            label="Show Current Image",
            initial_value=True
        )
        
        # Toggle for axes visibility
        self.toggle_axes = self.server.gui.add_checkbox(
            label="Show Coordinate Axes",
            initial_value=True
        )
        
        # Option to show difference image in current frustum
        self.show_diff = self.server.gui.add_checkbox(
            label="Show Difference Image",
            initial_value=False
        )
        
        # Store images for difference computation
        self.rgb_des = rgb_des
        self.rgb_cur = rgb_cur
        self.rgb_des_down = rgb_des_down
        self.rgb_cur_down = rgb_cur_down

        @self.toggle_des.on_update
        def _(event: viser.GuiEvent) -> None:
            self.frustum_des.visible = self.toggle_des.value
            self.axes_des.visible = self.toggle_des.value and self.toggle_axes.value

        @self.toggle_init.on_update
        def _(event: viser.GuiEvent) -> None:
            self.frustum_init.visible = self.toggle_init.value
            self.axes_init.visible = self.toggle_init.value and self.toggle_axes.value
            
        @self.toggle_cur.on_update
        def _(event: viser.GuiEvent) -> None:
            self.frustum_cur.visible = self.toggle_cur.value
            self.axes_cur.visible = self.toggle_cur.value and self.toggle_axes.value
            
        @self.toggle_axes.on_update
        def _(event: viser.GuiEvent) -> None:
            # Update all axes visibility based on their parent frustum visibility
            self.axes_des.visible = self.toggle_des.value and self.toggle_axes.value
            self.axes_init.visible = self.toggle_init.value and self.toggle_axes.value
            self.axes_cur.visible = self.toggle_cur.value and self.toggle_axes.value
            
        @self.show_diff.on_update
        def _(event: viser.GuiEvent) -> None:
            # Update current frustum image based on checkbox
            self._update_frustum_image()

        # -------------------------
        # Gaussian splats (optional)
        # -------------------------
        self.splats_handle = None

        if means is not None:
            SH_C0 = 0.28209479177387814
            rgbs = np.clip(0.5 + SH_C0 * sh0[:, 0, :], 0.0, 1.0)
            Rs = tf.SO3(quats).as_matrix()
            covariances = np.einsum(
                "nij,njk,nlk->nil",
                Rs,
                np.eye(3)[None, :, :] * scales[:, None, :] ** 2,
                Rs,
            )

            self.toggle_splats = self.server.gui.add_checkbox(
                label="Show Splats",
                initial_value=False,
            )

            @self.toggle_splats.on_update
            def _(event: viser.GuiEvent) -> None:
                if self.toggle_splats.value and self.splats_handle is None:
                    self.splats_handle = self.server.scene.add_gaussian_splats(
                        "/splats",
                        centers=means,
                        rgbs=rgbs,
                        opacities=opacities[:, None],
                        covariances=covariances,
                    )
                elif not self.toggle_splats.value and self.splats_handle is not None:
                    self.splats_handle.remove()
                    self.splats_handle = None
        # -------------------------
        # Plotly: error
        # -------------------------
        self.iter_history = []
        self.error_history = []

        self.fig_error = go.Figure()
        self.fig_error.add_trace(
            go.Scatter(x=[], y=[], mode="lines+markers", name="||e||²")
        )
        self.fig_error.update_layout(
            title="Photometric Error",
            xaxis_title="Iteration",
            yaxis_title="||e||²",
            yaxis_type="log",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        self.error_plot = self.server.gui.add_plotly(self.fig_error, aspect=1.4)

        # -------------------------
        # Plotly: velocity
        # -------------------------
        self.vel_history = []
        self.vel_labels = ["vx", "vy", "vz", "wx", "wy", "wz"]

        self.fig_vel = go.Figure()
        for lbl in self.vel_labels:
            self.fig_vel.add_trace(
                go.Scatter(x=[], y=[], mode="lines", name=lbl)
            )

        self.fig_vel.update_layout(
            title="Camera Velocity",
            xaxis_title="Iteration",
            yaxis_title="Velocity",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        self.vel_plot = self.server.gui.add_plotly(self.fig_vel, aspect=1.4)
        
        # -------------------------
        # Scene recording
        # -------------------------
        self.serializer = self.server.get_scene_serializer()

        self.save_button = self.server.gui.add_button("Download Recording")

        @self.save_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if event.client is not None:
                data = self.serializer.serialize()
                event.client.send_file_download("recording.viser", data)

    def _update_frustum_image(self):
        """Update the image shown in the current frustum based on settings."""
        if self.show_diff.value:
            # Show difference image
            diff_image = np.clip(self.rgb_cur - self.rgb_des, 0.0, 1.0)
            frustum_image = diff_image[::self.downsample_factor, ::self.downsample_factor]
        else:
            # Show current image
            frustum_image = self.rgb_cur[::self.downsample_factor, ::self.downsample_factor]
        
        self.frustum_cur.image = frustum_image

    # --------------------------------------------------
    # Update method (called from VS loop)
    # --------------------------------------------------
    def update(self, iteration, cMo, rgb, error, velocity):
        """
        Update visualization.

        Args:
            iteration: int
            cMo: 4x4 world->camera pose
            rgb: (H,W,3) torch or numpy, float in [0,1] or [0,>1]
            error: float
            velocity: (6,) numpy
        """

        # Convert to numpy if needed
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb_vis = np.clip(rgb, 0.0, 1.0)
        
        # Store current image
        self.rgb_cur = rgb_vis

        # Update current frustum pose
        self.frustum_cur.wxyz = tf.SO3.from_matrix(cMo[:3, :3]).wxyz
        self.frustum_cur.position = tuple(cMo[:3, 3])
        
        # Update frustum image
        self._update_frustum_image()

        # -------------------------
        # Update error plot
        # -------------------------
        self.iter_history.append(iteration)
        self.error_history.append(error)
        self.fig_error.data[0].x = self.iter_history
        self.fig_error.data[0].y = self.error_history
        self.error_plot.figure = self.fig_error

        # -------------------------
        # Update velocity plot
        # -------------------------
        self.vel_history.append(velocity.copy())
        for i in range(6):
            self.fig_vel.data[i].x = self.iter_history
            self.fig_vel.data[i].y = [v[i] for v in self.vel_history]
        self.vel_plot.figure = self.fig_vel
        
        # Record for scene serialization
        self.serializer.insert_sleep(1.0 / 30.0)  # 30 FPS

    def save(self, path="recording.viser"):
        data = self.serializer.serialize()
        Path(path).write_bytes(data)
        print(f"[INFO] Saved Viser scene recording to: {path}")
