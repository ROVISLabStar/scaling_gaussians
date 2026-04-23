"""
Viser-based viewer for Visual Servoing on Gaussian Splatting.
Supports PGM-VS only, PL-VS only, or side-by-side comparison,
with 3D camera trajectories, difference images, and live plots.
"""

import numpy as np
import viser
import plotly.graph_objects as go
from pathlib import Path
from viser import transforms as tf


class VsComparisonViewer:
    """
    Viewer for visual servoing experiments.
    Adapts to show only the active method(s) based on enable_pgm / enable_pl.
    """

    def __init__(
        self,
        rgb_des,
        cdMo,
        rgb_cur_pgm,
        cMo_pgm,
        rgb_cur_pl,
        cMo_pl,
        image_size,
        aspect_ratio,
        server_port=8080,
        image_scale=0.5,
        means=None,
        scales=None,
        quats=None,
        sh0=None,
        opacities=None,
        downsample_factor=1,
        axes_length=0.1,
        axes_radius=0.005,
        enable_pgm=True,
        enable_pl=True,
    ):
        self.W, self.H = image_size
        self.aspect_ratio = aspect_ratio
        self.downsample_factor = downsample_factor
        self.image_scale = image_scale
        self.enable_pgm = enable_pgm
        self.enable_pl = enable_pl

        # Ensure numpy
        if hasattr(rgb_des, 'cpu'):
            rgb_des = rgb_des.cpu().numpy()
        if hasattr(rgb_cur_pgm, 'cpu'):
            rgb_cur_pgm = rgb_cur_pgm.cpu().numpy()
        if hasattr(rgb_cur_pl, 'cpu'):
            rgb_cur_pl = rgb_cur_pl.cpu().numpy()

        rgb_des = np.clip(rgb_des, 0.0, 1.0)
        rgb_cur_pgm = np.clip(rgb_cur_pgm, 0.0, 1.0)
        rgb_cur_pl = np.clip(rgb_cur_pl, 0.0, 1.0)

        self.rgb_des = rgb_des
        self.rgb_ini = rgb_cur_pgm.copy()
        self.rgb_cur_pgm = rgb_cur_pgm
        self.rgb_cur_pl = rgb_cur_pl

        self._image_mode = "current"

        # 3D trajectory storage
        self._traj_pgm = [cMo_pgm[:3, 3].copy()] if enable_pgm else []
        self._traj_pl = [cMo_pl[:3, 3].copy()] if enable_pl else []
        self._traj_pgm_handle = None
        self._traj_pl_handle = None

        # Use the first available initial pose for the initial frustum
        cMo_init = cMo_pgm if enable_pgm else cMo_pl
        rgb_ini_np = rgb_cur_pgm if enable_pgm else rgb_cur_pl

        # -------------------------
        # Viser server
        # -------------------------
        self.server = viser.ViserServer(port=server_port)
        self.server.gui.configure_theme(control_layout="collapsible", control_width="large")
        print(f"[INFO] Open http://localhost:{server_port}")

        up_world = cdMo[:3, :3] @ np.array([0.0, -1.0, 0.0])
        up_world /= np.linalg.norm(up_world)
        self.server.scene.set_up_direction(tuple(up_world))

        fov = np.pi / 2
        rgb_des_down = rgb_des[::downsample_factor, ::downsample_factor]
        rgb_ini_down = rgb_ini_np[::downsample_factor, ::downsample_factor]

        # --- Desired frustum (green) ---
        self.frustum_des = self.server.scene.add_camera_frustum(
            "/frustums/desired",
            fov=fov, aspect=aspect_ratio, scale=image_scale,
            image=rgb_des_down,
            wxyz=tf.SO3.from_matrix(cdMo[:3, :3]).wxyz,
            position=tuple(cdMo[:3, 3]),
            color=(0, 255, 0),
        )
        self.server.scene.add_frame(
            "/frustums/desired/axes",
            axes_length=axes_length, axes_radius=axes_radius,
        )

        # --- Initial frustum (gray) ---
        self.frustum_init = self.server.scene.add_camera_frustum(
            "/frustums/initial",
            fov=fov, aspect=aspect_ratio, scale=image_scale,
            image=rgb_ini_down,
            wxyz=tf.SO3.from_matrix(cMo_init[:3, :3]).wxyz,
            position=tuple(cMo_init[:3, 3]),
            color=(180, 180, 180),
        )
        self.server.scene.add_frame(
            "/frustums/initial/axes",
            axes_length=axes_length, axes_radius=axes_radius,
        )

        # --- PGM current frustum (blue) ---
        self.frustum_pgm = None
        if enable_pgm:
            self.frustum_pgm = self.server.scene.add_camera_frustum(
                "/frustums/pgm_current",
                fov=fov, aspect=aspect_ratio, scale=image_scale,
                image=rgb_cur_pgm[::downsample_factor, ::downsample_factor],
                wxyz=tf.SO3.from_matrix(cMo_pgm[:3, :3]).wxyz,
                position=tuple(cMo_pgm[:3, 3]),
                color=(0, 100, 255),
            )
            self.server.scene.add_frame(
                "/frustums/pgm_current/axes",
                axes_length=axes_length, axes_radius=axes_radius,
            )

        # --- PL current frustum (red) ---
        self.frustum_pl = None
        if enable_pl:
            self.frustum_pl = self.server.scene.add_camera_frustum(
                "/frustums/pl_current",
                fov=fov, aspect=aspect_ratio, scale=image_scale,
                image=rgb_cur_pl[::downsample_factor, ::downsample_factor],
                wxyz=tf.SO3.from_matrix(cMo_pl[:3, :3]).wxyz,
                position=tuple(cMo_pl[:3, 3]),
                color=(255, 50, 50),
            )
            self.server.scene.add_frame(
                "/frustums/pl_current/axes",
                axes_length=axes_length, axes_radius=axes_radius,
            )

        # -------------------------
        # GUI controls
        # -------------------------
        # Title adapts to mode
        if enable_pgm and enable_pl:
            self.server.gui.add_markdown("## Comparison: PGM-VS vs PL-VS")
        elif enable_pgm:
            self.server.gui.add_markdown("## PGM-VS (Crombez et al. TRO 2019)")
        else:
            self.server.gui.add_markdown("## Photometric VS (Collewet & Marchand)")

        self.server.gui.add_markdown("### Visibility")
        self.toggle_des = self.server.gui.add_checkbox("Show Desired (green)", initial_value=True)
        self.toggle_init = self.server.gui.add_checkbox("Show Initial (gray)", initial_value=True)

        self.toggle_pgm = None
        self.toggle_traj_pgm = None
        if enable_pgm:
            self.toggle_pgm = self.server.gui.add_checkbox("Show PGM current (blue)", initial_value=True)
        self.toggle_pl = None
        self.toggle_traj_pl = None
        if enable_pl:
            self.toggle_pl = self.server.gui.add_checkbox("Show PL current (red)", initial_value=True)

        self.server.gui.add_markdown("### Trajectories")
        if enable_pgm:
            self.toggle_traj_pgm = self.server.gui.add_checkbox("Show PGM path (blue)", initial_value=True)
        if enable_pl:
            self.toggle_traj_pl = self.server.gui.add_checkbox("Show PL path (red)", initial_value=True)

        self.server.gui.add_markdown("### Frustum image mode")
        self.image_mode_dropdown = self.server.gui.add_dropdown(
            "Image display",
            options=["Current image", "Difference (|cur - des|)"],
            initial_value="Current image",
        )

        @self.toggle_des.on_update
        def _(event):
            self.frustum_des.visible = self.toggle_des.value

        @self.toggle_init.on_update
        def _(event):
            self.frustum_init.visible = self.toggle_init.value

        if enable_pgm:
            @self.toggle_pgm.on_update
            def _(event):
                self.frustum_pgm.visible = self.toggle_pgm.value

            @self.toggle_traj_pgm.on_update
            def _(event):
                if self._traj_pgm_handle is not None:
                    self._traj_pgm_handle.visible = self.toggle_traj_pgm.value

        if enable_pl:
            @self.toggle_pl.on_update
            def _(event):
                self.frustum_pl.visible = self.toggle_pl.value

            @self.toggle_traj_pl.on_update
            def _(event):
                if self._traj_pl_handle is not None:
                    self._traj_pl_handle.visible = self.toggle_traj_pl.value

        @self.image_mode_dropdown.on_update
        def _(event):
            if self.image_mode_dropdown.value == "Difference (|cur - des|)":
                self._image_mode = "difference"
            else:
                self._image_mode = "current"
            self._refresh_frustum_images()

        # -------------------------
        # Gaussian splats
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

            self.toggle_splats = self.server.gui.add_checkbox("Show Splats", initial_value=False)

            @self.toggle_splats.on_update
            def _(event):
                if self.toggle_splats.value and self.splats_handle is None:
                    self.splats_handle = self.server.scene.add_gaussian_splats(
                        "/splats",
                        centers=means, rgbs=rgbs,
                        opacities=opacities[:, None],
                        covariances=covariances,
                    )
                elif not self.toggle_splats.value and self.splats_handle is not None:
                    self.splats_handle.remove()
                    self.splats_handle = None

        # -------------------------
        # Plotly: error
        # -------------------------
        self.iter_pgm = []
        self.err_pgm = []
        self.iter_pl = []
        self.err_pl = []

        self.fig_error = go.Figure()
        if enable_pgm:
            self.fig_error.add_trace(
                go.Scatter(x=[], y=[], mode="lines", name="PGM-VS",
                           line=dict(color="blue"))
            )
        if enable_pl:
            self.fig_error.add_trace(
                go.Scatter(x=[], y=[], mode="lines", name="PL-VS",
                           line=dict(color="red"))
            )
        self.fig_error.update_layout(
            title="MSE Error",
            xaxis_title="Iteration",
            yaxis_title="MSE (||e||^2 / N)",
            yaxis_type="log",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        # Track trace indices for error plot
        self._err_pgm_idx = 0 if enable_pgm else None
        self._err_pl_idx = (1 if enable_pgm else 0) if enable_pl else None
        self.error_plot = self.server.gui.add_plotly(self.fig_error, aspect=1.4)

        # -------------------------
        # Plotly: velocities
        # -------------------------
        self.vel_pgm = []
        self.vel_pl = []
        self.vel_labels = ["vx", "vy", "vz", "wx", "wy", "wz"]

        self.vel_pgm_plot = None
        if enable_pgm:
            self.fig_vel_pgm = go.Figure()
            for lbl in self.vel_labels:
                self.fig_vel_pgm.add_trace(go.Scatter(x=[], y=[], mode="lines", name=lbl))
            self.fig_vel_pgm.update_layout(
                title="PGM-VS Velocity",
                xaxis_title="Iteration", yaxis_title="Velocity",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            self.vel_pgm_plot = self.server.gui.add_plotly(self.fig_vel_pgm, aspect=1.4)

        self.vel_pl_plot = None
        if enable_pl:
            self.fig_vel_pl = go.Figure()
            for lbl in self.vel_labels:
                self.fig_vel_pl.add_trace(go.Scatter(x=[], y=[], mode="lines", name=lbl))
            self.fig_vel_pl.update_layout(
                title="PL-VS Velocity",
                xaxis_title="Iteration", yaxis_title="Velocity",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            self.vel_pl_plot = self.server.gui.add_plotly(self.fig_vel_pl, aspect=1.4)

        # -------------------------
        # Plotly: lambda_g evolution (PGM only)
        # -------------------------
        self.lambda_history = []
        self.lambda_star_history = []
        self.lambda_plot = None

        if enable_pgm:
            self.fig_lambda = go.Figure()
            self.fig_lambda.add_trace(
                go.Scatter(x=[], y=[], mode="lines", name="lambda_g",
                           line=dict(color="blue"))
            )
            self.fig_lambda.add_trace(
                go.Scatter(x=[], y=[], mode="lines", name="lambda_g*",
                           line=dict(color="blue", dash="dash"))
            )
            self.fig_lambda.update_layout(
                title="PGM Extension Parameter",
                xaxis_title="Iteration", yaxis_title="lambda_g",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            self.lambda_plot = self.server.gui.add_plotly(self.fig_lambda, aspect=1.4)

        # Scene recording
        self.serializer = self.server.get_scene_serializer()
        self.save_button = self.server.gui.add_button("Download Recording")

        @self.save_button.on_click
        def _(event):
            if event.client is not None:
                data = self.serializer.serialize()
                event.client.send_file_download("recording.viser", data)

    # --------------------------------------------------
    # Trajectory helpers
    # --------------------------------------------------
    def _update_trajectory_pgm(self):
        if not self.enable_pgm or len(self._traj_pgm) < 2:
            return
        pts = np.array(self._traj_pgm, dtype=np.float32)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        if self._traj_pgm_handle is not None:
            self._traj_pgm_handle.remove()
        self._traj_pgm_handle = self.server.scene.add_line_segments(
            "/trajectories/pgm",
            points=segments,
            colors=(0, 100, 255),
            line_width=3.0,
        )
        self._traj_pgm_handle.visible = self.toggle_traj_pgm.value

    def _update_trajectory_pl(self):
        if not self.enable_pl or len(self._traj_pl) < 2:
            return
        pts = np.array(self._traj_pl, dtype=np.float32)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        if self._traj_pl_handle is not None:
            self._traj_pl_handle.remove()
        self._traj_pl_handle = self.server.scene.add_line_segments(
            "/trajectories/pl",
            points=segments,
            colors=(255, 50, 50),
            line_width=3.0,
        )
        self._traj_pl_handle.visible = self.toggle_traj_pl.value

    # --------------------------------------------------
    # Frustum image helpers
    # --------------------------------------------------
    def _compute_frustum_image(self, rgb):
        ds = self.downsample_factor
        if self._image_mode == "difference":
            diff = np.abs(rgb - self.rgb_des)
            diff_vis = np.clip(diff * 3.0, 0.0, 1.0)
            return diff_vis[::ds, ::ds]
        else:
            return rgb[::ds, ::ds]

    def _refresh_frustum_images(self):
        if self.frustum_pgm is not None:
            self.frustum_pgm.image = self._compute_frustum_image(self.rgb_cur_pgm)
        if self.frustum_pl is not None:
            self.frustum_pl.image = self._compute_frustum_image(self.rgb_cur_pl)

    # --------------------------------------------------
    # Update methods (called from VS loop)
    # --------------------------------------------------
    def update_pgm(self, iteration, cMo, rgb, error, velocity, lambda_g=None, lambda_g_star=None):
        if not self.enable_pgm:
            return
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb_vis = np.clip(rgb, 0.0, 1.0)
        self.rgb_cur_pgm = rgb_vis

        self.frustum_pgm.wxyz = tf.SO3.from_matrix(cMo[:3, :3]).wxyz
        self.frustum_pgm.position = tuple(cMo[:3, 3])
        self.frustum_pgm.image = self._compute_frustum_image(rgb_vis)

        self._traj_pgm.append(cMo[:3, 3].copy())

        self.iter_pgm.append(iteration)
        self.err_pgm.append(error)
        self.fig_error.data[self._err_pgm_idx].x = self.iter_pgm
        self.fig_error.data[self._err_pgm_idx].y = self.err_pgm

        self.vel_pgm.append(velocity.copy())
        for i in range(6):
            self.fig_vel_pgm.data[i].x = self.iter_pgm
            self.fig_vel_pgm.data[i].y = [v[i] for v in self.vel_pgm]

        if lambda_g is not None:
            self.lambda_history.append(lambda_g)
            self.fig_lambda.data[0].x = self.iter_pgm
            self.fig_lambda.data[0].y = self.lambda_history
        if lambda_g_star is not None:
            self.lambda_star_history.append(lambda_g_star)
            self.fig_lambda.data[1].x = self.iter_pgm
            self.fig_lambda.data[1].y = self.lambda_star_history

    def update_pl(self, iteration, cMo, rgb, error, velocity):
        if not self.enable_pl:
            return
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb_vis = np.clip(rgb, 0.0, 1.0)
        self.rgb_cur_pl = rgb_vis

        self.frustum_pl.wxyz = tf.SO3.from_matrix(cMo[:3, :3]).wxyz
        self.frustum_pl.position = tuple(cMo[:3, 3])
        self.frustum_pl.image = self._compute_frustum_image(rgb_vis)

        self._traj_pl.append(cMo[:3, 3].copy())

        self.iter_pl.append(iteration)
        self.err_pl.append(error)
        self.fig_error.data[self._err_pl_idx].x = self.iter_pl
        self.fig_error.data[self._err_pl_idx].y = self.err_pl

        self.vel_pl.append(velocity.copy())
        for i in range(6):
            self.fig_vel_pl.data[i].x = self.iter_pl
            self.fig_vel_pl.data[i].y = [v[i] for v in self.vel_pl]

    def refresh_plots(self, iteration=0):
        self.error_plot.figure = self.fig_error
        if self.vel_pgm_plot is not None:
            self.vel_pgm_plot.figure = self.fig_vel_pgm
        if self.vel_pl_plot is not None:
            self.vel_pl_plot.figure = self.fig_vel_pl
        if self.lambda_plot is not None:
            self.lambda_plot.figure = self.fig_lambda
        if iteration % 5 == 0 or iteration < 10:
            self._update_trajectory_pgm()
            self._update_trajectory_pl()
        self.serializer.insert_sleep(1.0 / 30.0)

    def save(self, path="recording.viser"):
        data = self.serializer.serialize()
        Path(path).write_bytes(data)
        print(f"[INFO] Saved recording to: {path}")
