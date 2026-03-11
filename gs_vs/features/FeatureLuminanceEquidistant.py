import torch
import torch.nn.functional as F
import math


class FeatureLuminanceEquidistant:
    """
    Photometric visual feature for equidistant (f·θ) projection model.

    Equidistant projection:
        u = fx * θ * cos(φ) + u0
        v = fy * θ * sin(φ) + v0

    where θ = arctan(r_xy / Z) is the angle from the optical axis,
          φ = atan2(Y, X) is the azimuth,
          r_xy = √(X² + Y²).

    The radial law is r = f·θ, in contrast to r = f·tan(θ) for pinhole.

    Back-projection (pixel → 3D):
        Given pixel (u, v) and depth Z along the optical axis:
        1. Compute r_pix = √((u-u0)² + (v-v0)²)
        2. θ = r_pix / f   (equidistant inverse)
        3. φ = atan2(v-v0, u-u0)
        4. X = Z * tan(θ) * cos(φ),  Y = Z * tan(θ) * sin(φ)

    Note: the depth map from Gaussian Splatting gives Euclidean distance ρ,
    not Z along the optical axis. When using ρ:
        X = ρ * sin(θ) * cos(φ),  Y = ρ * sin(θ) * sin(φ),  Z = ρ * cos(θ)
    """

    def __init__(self, border=10, device="cuda", fov_max=180.0):
        """
        Args:
            border: Number of pixels to ignore at image borders
            device: Computation device ('cuda' or 'cpu')
            fov_max: Maximum field of view in degrees (for validity checking)
        """
        self.device = torch.device(device)
        self.border = border
        self.fov_max = torch.tensor(fov_max * math.pi / 180.0, device=self.device)

        # Camera parameters (to be set)
        self.cam = None
        self.nbr = None
        self.nbc = None

        # Feature data
        self.X = None       # 3D X in camera frame
        self.Y = None       # 3D Y in camera frame
        self.Z = None       # 3D Z in camera frame (along optical axis)
        self.I = None        # Intensity values
        self.Ix = None       # Image gradient ∂I/∂u (pixel units)
        self.Iy = None       # Image gradient ∂I/∂v (pixel units)
        self.s = None        # Feature vector (intensities)

        # Angular coordinates
        self.theta = None    # Angle from optical axis
        self.phi = None      # Azimuth angle

        # Validity
        self.valid_indices = None
        self._error = None
        self.dim_s = 0

        # Original image (for reference / interpolation)
        self.image_original = None

    def init(self, height, width):
        """Initialize image dimensions."""
        self.nbr = int(height)
        self.nbc = int(width)

    def setCameraParameters(self, cam):
        """
        Set camera intrinsic parameters.

        Args:
            cam: Object with attributes px (fx), py (fy), u0, v0
        """
        self.cam = cam

    # ------------------------------------------------------------------
    # Image gradients
    # ------------------------------------------------------------------

    def compute_gradients_image_plane(self, image):
        """
        Compute image gradients using 7-tap Gaussian derivative filter.

        Args:
            image: [H, W] grayscale image tensor

        Returns:
            Ix, Iy: Image gradients ∂I/∂u and ∂I/∂v (pixel units)
        """
        coeffs = torch.tensor(
            [112.0, 913.0, 2047.0], dtype=torch.float32, device=self.device
        ) / 8418.0

        kernel = torch.zeros(7, dtype=torch.float32, device=self.device)
        kernel[0], kernel[1], kernel[2] = -coeffs[2], -coeffs[1], -coeffs[0]
        kernel[4], kernel[5], kernel[6] =  coeffs[0],  coeffs[1],  coeffs[2]

        kernel_x = kernel.view(1, 1, 1, 7)
        kernel_y = kernel.view(1, 1, 7, 1)

        img4d = image.unsqueeze(0).unsqueeze(0)
        pad_x = F.pad(img4d, (3, 3, 0, 0), mode='reflect')
        pad_y = F.pad(img4d, (0, 0, 3, 3), mode='reflect')

        Ix = F.conv2d(pad_x, kernel_x)[0, 0]
        Iy = F.conv2d(pad_y, kernel_y)[0, 0]
        return Ix, Iy

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build visual features from image and depth map.

        Back-projection uses the equidistant inverse:
            θ = r_pix / f,   φ = atan2(v - v0, u - u0)
        then reconstructs 3D coordinates from (θ, φ, depth).

        The depth from Gaussian Splatting is Euclidean distance ρ = ||X||,
        so we use:
            X = ρ sin(θ) cos(φ),  Y = ρ sin(θ) sin(φ),  Z = ρ cos(θ)

        If your depth is Z along the optical axis instead, change the
        back-projection accordingly (see comments in code).

        Args:
            image: [H, W] grayscale image tensor
            depth: [H, W] depth map (Euclidean distance ρ, or None for unit)
            mask:  [H, W] boolean mask (True = valid pixel)
        """
        assert isinstance(image, torch.Tensor), "Input must be a torch.Tensor"
        assert self.cam is not None, "Call setCameraParameters() first"

        image = image.to(self.device).float()
        self.image_original = image

        fx = torch.tensor(self.cam.px, device=self.device).float()
        fy = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        # Use average focal length for the equidistant radial model r = f·θ
        # (equidistant assumes a single radial focal length; fx ≈ fy for
        #  a well-calibrated fisheye. We use the average for back-projection
        #  and the separate fx, fy for the forward Jacobian.)
        f_avg = (fx + fy) * 0.5

        # Image gradients (pixel units)
        Ix, Iy = self.compute_gradients_image_plane(image)

        # Border crop
        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        I_valid  = image[i0:i1, j0:j1]
        Ix_valid = Ix[i0:i1, j0:j1]
        Iy_valid = Iy[i0:i1, j0:j1]

        # Pixel coordinate grid
        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device, dtype=torch.float32),
            torch.arange(i0, i1, device=self.device, dtype=torch.float32),
            indexing='xy'
        )

        # ---- Equidistant back-projection ----
        # Pixel displacement from principal point
        du = jj - u0       # Δu
        dv = ii - v0       # Δv

        # Radial distance in pixel space
        r_pix = torch.sqrt(du**2 + dv**2 + 1e-12)

        # Equidistant inverse: θ = r_pix / f
        theta = r_pix / f_avg

        # Azimuth
        phi = torch.atan2(dv, du)

        # Handle depth
        if depth is not None:
            depth_map = depth.to(self.device).float()
            rho_valid = depth_map[i0:i1, j0:j1]
            # Clean invalid depth values
            rho_valid = torch.where(
                (rho_valid <= 1e-6) | torch.isnan(rho_valid) | torch.isinf(rho_valid),
                torch.tensor(1.0, device=self.device),
                rho_valid
            )
        else:
            rho_valid = torch.ones_like(I_valid)

        # 3D coordinates from (θ, φ, ρ)
        # If depth is Euclidean distance ρ = ||X||:
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        X_3d = rho_valid * sin_theta * cos_phi
        Y_3d = rho_valid * sin_theta * sin_phi
        Z_3d = rho_valid * cos_theta

        # ---- If your depth map gives Z along optical axis instead of ρ: ----
        # Uncomment the following and comment out the block above:
        # Z_3d = rho_valid   # depth IS Z
        # tan_theta = torch.tan(theta)
        # X_3d = Z_3d * tan_theta * cos_phi
        # Y_3d = Z_3d * tan_theta * sin_phi

        # Flatten
        X_flat     = X_3d.reshape(-1)
        Y_flat     = Y_3d.reshape(-1)
        Z_flat     = Z_3d.reshape(-1)
        I_flat     = I_valid.reshape(-1)
        Ix_flat    = Ix_valid.reshape(-1)
        Iy_flat    = Iy_valid.reshape(-1)
        theta_flat = theta.reshape(-1)
        phi_flat   = phi.reshape(-1)

        # ---- Validity filtering ----
        valid = (
            (Z_flat > 1e-8)                    # In front of camera
            & (theta_flat <= self.fov_max)     # Within FoV
            & (theta_flat > 1e-8)              # Not exactly on optical axis (avoid 0/0)
            & torch.isfinite(X_flat)
            & torch.isfinite(Y_flat)
            & torch.isfinite(Z_flat)
        )

        if mask is not None:
            mask_flat = mask.to(self.device).bool()[i0:i1, j0:j1].reshape(-1)
            valid = valid & mask_flat

        # Store valid data
        self.valid_indices = valid
        self.X     = X_flat[valid]
        self.Y     = Y_flat[valid]
        self.Z     = Z_flat[valid]
        self.I     = I_flat[valid]
        self.Ix    = Ix_flat[valid]
        self.Iy    = Iy_flat[valid]
        self.theta = theta_flat[valid]
        self.phi   = phi_flat[valid]

        self.s = self.I.clone()
        self.dim_s = self.X.shape[0]

        assert torch.isfinite(self.Z).all(), "NaN/Inf in Z after filtering"

    # ------------------------------------------------------------------
    # Interaction matrix
    # ------------------------------------------------------------------

    def interaction(self):
        """
        Photometric interaction matrix for equidistant projection.

        Derivation (chain rule, consistent with paper Eq. 4):

            L_I = -∇I(u)  ·  J_eq  ·  L_X

        where:
          ∇I = (∂I/∂u, ∂I/∂v)  ← image gradient (1×2 per pixel)

          J_eq = ∂u/∂(θ,φ) · ∂(θ,φ)/∂X   ← equidistant projection Jacobian (2×3)

          L_X = standard 3D point interaction matrix (3×6)

        The Jacobian J_eq uses separate fx, fy:
          ∂u/∂(θ,φ) = [[fx cosφ,  -fx θ sinφ],
                        [fy sinφ,   fy θ cosφ]]

          ∂θ/∂X = XZ/(r_xy ρ²),  ∂θ/∂Y = YZ/(r_xy ρ²),  ∂θ/∂Z = -r_xy/ρ²
          ∂φ/∂X = -Y/r_xy²,      ∂φ/∂Y = X/r_xy²,        ∂φ/∂Z = 0

        Returns:
            L_I: (N, 6) interaction matrix
        """
        N = self.X.shape[0]
        X, Y, Z = self.X, self.Y, self.Z
        theta = self.theta
        phi = self.phi

        fx = torch.tensor(self.cam.px, device=self.device).float()
        fy = torch.tensor(self.cam.py, device=self.device).float()

        # Intermediate quantities
        r_xy_sq = X**2 + Y**2
        r_xy = torch.sqrt(r_xy_sq + 1e-12)         # √(X²+Y²)
        rho_sq = r_xy_sq + Z**2
        rho = torch.sqrt(rho_sq + 1e-12)            # ||X||

        # ---- ∂(θ,φ)/∂X  (2×3 per pixel) ----
        # ∂θ/∂X_i
        inv_rxy_rhosq = 1.0 / (r_xy * rho_sq + 1e-12)
        dtheta_dX =  X * Z * inv_rxy_rhosq
        dtheta_dY =  Y * Z * inv_rxy_rhosq
        dtheta_dZ = -r_xy / (rho_sq + 1e-12)

        # ∂φ/∂X_i
        inv_rxy_sq = 1.0 / (r_xy_sq + 1e-12)
        dphi_dX = -Y * inv_rxy_sq
        dphi_dY =  X * inv_rxy_sq
        # dphi_dZ = 0 (not stored)

        # ---- ∂u/∂(θ,φ) · ∂(θ,φ)/∂X  →  J_eq (2×3) ----
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        # Row 1: ∂u/∂X = fx (cosφ · ∂θ/∂X  -  θ sinφ · ∂φ/∂X)
        du_dX = fx * (cos_phi * dtheta_dX - theta * sin_phi * dphi_dX)
        du_dY = fx * (cos_phi * dtheta_dY - theta * sin_phi * dphi_dY)
        du_dZ = fx * (cos_phi * dtheta_dZ)  # dphi_dZ = 0

        # Row 2: ∂v/∂X = fy (sinφ · ∂θ/∂X  +  θ cosφ · ∂φ/∂X)
        dv_dX = fy * (sin_phi * dtheta_dX + theta * cos_phi * dphi_dX)
        dv_dY = fy * (sin_phi * dtheta_dY + theta * cos_phi * dphi_dY)
        dv_dZ = fy * (sin_phi * dtheta_dZ)  # dphi_dZ = 0

        # Assemble J_proj  (N, 2, 3)
        J_proj = torch.zeros((N, 2, 3), device=self.device)
        J_proj[:, 0, 0] = du_dX
        J_proj[:, 0, 1] = du_dY
        J_proj[:, 0, 2] = du_dZ
        J_proj[:, 1, 0] = dv_dX
        J_proj[:, 1, 1] = dv_dY
        J_proj[:, 1, 2] = dv_dZ

        # ---- L_X: 3D point interaction matrix (N, 3, 6) ----
        # [[-1  0  0   0  -Z   Y ]
        #  [ 0 -1  0   Z   0  -X ]
        #  [ 0  0 -1  -Y   X   0 ]]
        L_X = torch.zeros((N, 3, 6), device=self.device)
        L_X[:, 0, 0] = -1.0
        L_X[:, 1, 1] = -1.0
        L_X[:, 2, 2] = -1.0
        L_X[:, 0, 3] =  0.0;  L_X[:, 0, 4] = -Z;  L_X[:, 0, 5] =  Y
        L_X[:, 1, 3] =  Z;    L_X[:, 1, 4] =  0.0; L_X[:, 1, 5] = -X
        L_X[:, 2, 3] = -Y;    L_X[:, 2, 4] =  X;   L_X[:, 2, 5] =  0.0

        # ---- Combine: L_u = J_proj · L_X  (N, 2, 6) ----
        L_u = torch.bmm(J_proj, L_X)

        # ---- Photometric interaction: L_I = -∇I^T · L_u  (N, 6) ----
        grad_I = torch.stack([self.Ix, self.Iy], dim=1)  # (N, 2)
        L_I = -torch.bmm(grad_I.unsqueeze(1), L_u).squeeze(1)  # (N, 6)

        return L_I

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def error(self, s_star):
        """
        Photometric error e = I - I*.

        Args:
            s_star: Desired feature object (same class, same valid pixel count)

        Returns:
            error: (N,) tensor
        """
        assert self.I.shape == s_star.I.shape, \
            f"Feature dimension mismatch: {self.I.shape} vs {s_star.I.shape}"
        self._error = self.I - s_star.I
        return self._error

    # ------------------------------------------------------------------
    # Weighted interaction (M-estimators)
    # ------------------------------------------------------------------

    def weighted_interaction(self, s_star, estimator="tukey", param=4.685):
        """
        Compute weighted interaction matrix using M-estimators.

        Args:
            s_star: Desired features
            estimator: "tukey" or "huber"
            param: Tuning constant

        Returns:
            L_weighted, error_weighted, weights
        """
        if self._error is None:
            self.error(s_star)

        error = self._error.view(-1)

        if estimator == "tukey":
            e2 = error ** 2
            weights = torch.zeros_like(error)
            inside = e2 < param ** 2
            weights[inside] = (1.0 - e2[inside] / param**2) ** 2

        elif estimator == "huber":
            abs_e = torch.abs(error)
            weights = torch.ones_like(error)
            outside = abs_e > param
            weights[outside] = param / abs_e[outside]

        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        L = self.interaction()
        L_weighted = weights.view(-1, 1) * L
        error_weighted = weights * error

        return L_weighted, error_weighted, weights

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        """Reset stored error between iterations."""
        self._error = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def back_project_to_pixel(self):
        """
        Forward-project stored 3D points back to pixel coordinates.
        Useful for verifying the back-projection round-trip.

        Returns:
            u, v: Pixel coordinates (tensors)
        """
        if self.X is None:
            return None, None

        fx = torch.tensor(self.cam.px, device=self.device).float()
        fy = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        u = fx * self.theta * torch.cos(self.phi) + u0
        v = fy * self.theta * torch.sin(self.phi) + v0
        return u, v

    def get_spherical_coordinates(self):
        """
        Return stored spherical coordinates for visualization.

        Returns:
            theta, phi, rho
        """
        if self.X is None:
            return None, None, None
        rho = torch.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        return self.theta, self.phi, rho
