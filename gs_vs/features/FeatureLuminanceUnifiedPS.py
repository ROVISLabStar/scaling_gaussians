import torch
import torch.nn.functional as F
import math


class FeatureLuminanceUnifiedPS:
    """
    Pure Spherical Photometric Visual Servoing (PS-VS).

    From Caron, Marchand, Mouaddib, Autonomous Robots 2013, Section 3.3.

    The feature is the image intensity indexed by spherical coordinates
    S = (φ, θ) on the equivalence sphere of the unified camera model:
        φ = arccos(Z_S)    (elevation from optical axis, 0 at pole)
        θ = atan2(Y_S, X_S) (azimuth)

    The photometric interaction matrix is:
        L_IS(S) = -∇I_S^T  L_S     (Eq. 29)

    where ∇I_S = (∂I_S/∂φ, ∂I_S/∂θ) are gradients in spherical coordinates
    computed via finite differences, and L_S is the geometric interaction
    matrix (Eq. 30):

        L_S = [[-cθcφ/ρ   -sθcφ/ρ    sφ/ρ    sθ      -cθ       0  ]
               [ sθ/(ρsφ) -cθ/(ρsφ)   0    cθcφ/sφ  sθcφ/sφ   -1  ]]

    Singularity: sinφ = 0  (at the optical axis, i.e., principal point).

    NOTE on depth: In the unified model, ρ = ||X|| (Euclidean distance).
    GSplat's "expected depth" (ED) is the rendered Euclidean depth.
    """

    def __init__(self, border=10, device="cuda", xi=1.635):
        self.device = torch.device(device)
        self.border = border
        self.xi = xi

        self.cam = None
        self.nbr = None
        self.nbc = None
        self.dim_s = 0

        # Stored data
        self.xs = None      # Normalized image x
        self.ys = None      # Normalized image y
        self.I = None        # Intensities
        self.Zs = None       # Depth (ρ = Euclidean distance)
        self.s = None
        self._error = None

        # Spherical coordinates
        self.phi = None      # arccos(Z_S) — elevation
        self.theta = None    # atan2(Y_S, X_S) — azimuth
        self.XS = None       # (N, 3) unit sphere points

        # For interpolation
        self.image_original = None

    def init(self, height, width):
        self.nbr = int(height)
        self.nbc = int(width)

    def setCameraParameters(self, cam, xi=1.635):
        """
        Set camera parameters
        
        Args:
            cam: Camera object with px, py, u0, v0
            xi: ξ parameter for unified model (default 1.635 for fisheye)
        """
        self.cam = cam
        if xi is not None:
            self.xi = xi
        else:
            self.xi = 1.635  # Default unified fisheye (matching TorchComplete line 91)

    # ------------------------------------------------------------------
    # Gradient filter (same 7-tap Gaussian derivative as other classes)
    # ------------------------------------------------------------------

    def _gaussian_derivative_kernel(self):
        coeffs = torch.tensor(
            [112.0, 913.0, 2047.0], dtype=torch.float32, device=self.device
        ) / 8418.0
        kernel = torch.zeros(7, dtype=torch.float32, device=self.device)
        kernel[0], kernel[1], kernel[2] = -coeffs[2], -coeffs[1], -coeffs[0]
        kernel[4], kernel[5], kernel[6] =  coeffs[0],  coeffs[1],  coeffs[2]
        return kernel

    # ------------------------------------------------------------------
    # Backprojection: image plane → unit sphere (UCM inverse, Eq. 5)
    # ------------------------------------------------------------------

    def _backproject_to_sphere(self, x, y):
        """
        Unified model inverse: normalized coords (x,y) → unit sphere X_S.
        Eq. (5) of Caron et al. 2013.
        """
        xi = self.xi
        r2 = x**2 + y**2
        alpha = torch.sqrt(torch.clamp(1.0 + (1.0 - xi**2) * r2, min=1e-8))
        denom = r2 + 1.0

        XS = torch.stack([
            (xi + alpha) * x / denom,
            (xi + alpha) * y / denom,
            (xi + alpha) / denom - xi,
        ], dim=-1)

        # Normalize to unit sphere
        XS_norm = torch.norm(XS, dim=-1, keepdim=True)
        XS = XS / (XS_norm + 1e-10)
        return XS

    # ------------------------------------------------------------------
    # Forward projection: unit sphere → normalized image plane
    # ------------------------------------------------------------------

    def _sphere_to_image(self, XS):
        """
        Unified model forward: X_S → (x, y).
        x = X_S / (Z_S + ξ),  y = Y_S / (Z_S + ξ)
        """
        xi = self.xi
        denom = XS[..., 2] + xi + 1e-10
        x = XS[..., 0] / denom
        y = XS[..., 1] / denom
        return x, y

    # ------------------------------------------------------------------
    # Bilinear interpolation on stored image
    # ------------------------------------------------------------------

    def _interpolate_image(self, u, v):
        """Bilinear interpolation at sub-pixel (u, v) coordinates."""
        H, W = self.image_original.shape
        u_f = u.float()
        v_f = v.float()

        u0 = torch.floor(u_f).long().clamp(0, W - 1)
        u1 = (u0 + 1).clamp(0, W - 1)
        v0 = torch.floor(v_f).long().clamp(0, H - 1)
        v1 = (v0 + 1).clamp(0, H - 1)

        wu = u_f - u0.float()
        wv = v_f - v0.float()

        I_interp = (
            (1 - wu) * (1 - wv) * self.image_original[v0, u0]
            + wu * (1 - wv) * self.image_original[v0, u1]
            + (1 - wu) * wv * self.image_original[v1, u0]
            + wu * wv * self.image_original[v1, u1]
        )
        return I_interp

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build PS-VS features from image and depth.

        Computes normalized coords, backprojects to unit sphere,
        converts to (φ, θ), stores everything.
        """
        assert isinstance(image, torch.Tensor)
        assert self.cam is not None

        image = image.to(self.device).float()
        self.image_original = image

        px = self.cam.px
        py = self.cam.py
        u0 = self.cam.u0
        v0 = self.cam.v0

        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        I_valid = image[i0:i1, j0:j1]

        # Pixel grids
        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device, dtype=torch.float32),
            torch.arange(i0, i1, device=self.device, dtype=torch.float32),
            indexing='xy'
        )

        # Normalized image coords
        x = (jj - u0) / px
        y = (ii - v0) / py

        # Depth
        if depth is not None:
            depth_map = depth.to(self.device).float()
            Z_valid = depth_map[i0:i1, j0:j1]
            Z_valid = torch.where(
                (Z_valid <= 1e-6) | torch.isnan(Z_valid) | torch.isinf(Z_valid),
                torch.tensor(1.0, device=self.device), Z_valid
            )
        else:
            Z_valid = torch.ones_like(I_valid)

        # Flatten
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        I_flat = I_valid.reshape(-1)
        Z_flat = Z_valid.reshape(-1)

        # UCM validity
        r2 = x_flat**2 + y_flat**2
        arg = 1.0 + (1.0 - self.xi**2) * r2
        valid = arg > 0

        if mask is not None:
            mask_flat = mask.to(self.device).bool()[i0:i1, j0:j1].reshape(-1)
            valid = valid & mask_flat

        # Store valid pixels
        self.xs = x_flat[valid]
        self.ys = y_flat[valid]
        self.I = I_flat[valid]
        self.Zs = Z_flat[valid]
        self.s = self.I.clone()

        # Backproject to sphere
        self.XS = self._backproject_to_sphere(self.xs, self.ys)  # (N, 3)

        # Spherical coordinates (Eq. 2)
        # φ = arccos(Z_S)  — elevation from north pole (optical axis)
        # θ = atan2(Y_S, X_S) — azimuth
        self.phi = torch.acos(torch.clamp(self.XS[:, 2], -1.0, 1.0))
        self.theta = torch.atan2(self.XS[:, 1], self.XS[:, 0])

        # Filter out singularity: sinφ ≈ 0 (at optical axis)
        sin_phi = torch.sin(self.phi)
        singular = torch.abs(sin_phi) < 1e-4
        non_singular = ~singular

        if non_singular.sum() < self.xs.shape[0]:
            self.xs = self.xs[non_singular]
            self.ys = self.ys[non_singular]
            self.I = self.I[non_singular]
            self.Zs = self.Zs[non_singular]
            self.s = self.I.clone()
            self.XS = self.XS[non_singular]
            self.phi = self.phi[non_singular]
            self.theta = self.theta[non_singular]

        self.dim_s = self.I.shape[0]
        assert torch.isfinite(self.Zs).all()

    # ------------------------------------------------------------------
    # Pure spherical gradients via central finite differences
    # ------------------------------------------------------------------

    def _compute_spherical_gradients(self):
        """
        Compute ∂I_S/∂φ and ∂I_S/∂θ via central finite differences
        on the equivalence sphere. (Section 3.3.3 of Caron et al.)

        For each pixel, we perturb φ ± Δφ and θ ± Δθ, project the
        resulting sphere point back to the image, interpolate intensity,
        and compute the gradient.
        """
        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0_val = torch.tensor(self.cam.u0, device=self.device).float()
        v0_val = torch.tensor(self.cam.v0, device=self.device).float()

        phi = self.phi       # (N,)
        theta = self.theta   # (N,)

        # Step size (from Eq. 25 adapted to pure spherical)
        delta = 0.01  # radians — small step for FD

        # ---- ∂I_S/∂φ via central difference ----
        # φ + Δφ
        phi_p = phi + delta
        XS_p = torch.stack([
            torch.sin(phi_p) * torch.cos(theta),
            torch.sin(phi_p) * torch.sin(theta),
            torch.cos(phi_p),
        ], dim=1)
        x_p, y_p = self._sphere_to_image(XS_p)
        u_p = x_p * px + u0_val
        v_p = y_p * py + v0_val
        I_phi_p = self._interpolate_image(u_p, v_p)

        # φ - Δφ
        phi_m = phi - delta
        XS_m = torch.stack([
            torch.sin(phi_m) * torch.cos(theta),
            torch.sin(phi_m) * torch.sin(theta),
            torch.cos(phi_m),
        ], dim=1)
        x_m, y_m = self._sphere_to_image(XS_m)
        u_m = x_m * px + u0_val
        v_m = y_m * py + v0_val
        I_phi_m = self._interpolate_image(u_m, v_m)

        dI_dphi = (I_phi_p - I_phi_m) / (2.0 * delta)

        # ---- ∂I_S/∂θ via central difference ----
        # θ + Δθ
        theta_p = theta + delta
        XS_p = torch.stack([
            torch.sin(phi) * torch.cos(theta_p),
            torch.sin(phi) * torch.sin(theta_p),
            torch.cos(phi),
        ], dim=1)
        x_p, y_p = self._sphere_to_image(XS_p)
        u_p = x_p * px + u0_val
        v_p = y_p * py + v0_val
        I_theta_p = self._interpolate_image(u_p, v_p)

        # θ - Δθ
        theta_m = theta - delta
        XS_m = torch.stack([
            torch.sin(phi) * torch.cos(theta_m),
            torch.sin(phi) * torch.sin(theta_m),
            torch.cos(phi),
        ], dim=1)
        x_m, y_m = self._sphere_to_image(XS_m)
        u_m = x_m * px + u0_val
        v_m = y_m * py + v0_val
        I_theta_m = self._interpolate_image(u_m, v_m)

        dI_dtheta = (I_theta_p - I_theta_m) / (2.0 * delta)

        return dI_dphi, dI_dtheta

    # ------------------------------------------------------------------
    # Geometric interaction matrix L_S (Eq. 30)
    # ------------------------------------------------------------------

    def _compute_LS(self):
        """
        Pure spherical geometric interaction matrix, Eq. (30):

        L_S = [[-cθcφ/ρ   -sθcφ/ρ    sφ/ρ    sθ      -cθ       0  ]
               [ sθ/(ρsφ) -cθ/(ρsφ)   0    cθcφ/sφ  sθcφ/sφ   -1  ]]

        Returns:
            LS: (N, 2, 6)
        """
        N = self.dim_s

        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)
        cos_phi = torch.cos(self.phi)
        sin_phi = torch.sin(self.phi)

        rho = self.Zs  # Euclidean distance
        rho_inv = 1.0 / (rho + 1e-10)

        # Safe division by sin(φ)
        sin_phi_safe = torch.where(
            torch.abs(sin_phi) < 1e-6,
            torch.tensor(1e-6, device=self.device),
            sin_phi
        )

        LS = torch.zeros((N, 2, 6), device=self.device)

        # Row 0: ∂φ/∂r  (elevation)
        LS[:, 0, 0] = -cos_theta * cos_phi * rho_inv
        LS[:, 0, 1] = -sin_theta * cos_phi * rho_inv
        LS[:, 0, 2] =  sin_phi * rho_inv
        LS[:, 0, 3] =  sin_theta
        LS[:, 0, 4] = -cos_theta
        LS[:, 0, 5] =  0.0

        # Row 1: ∂θ/∂r  (azimuth)
        LS[:, 1, 0] =  sin_theta * rho_inv / sin_phi_safe
        LS[:, 1, 1] = -cos_theta * rho_inv / sin_phi_safe
        LS[:, 1, 2] =  0.0
        LS[:, 1, 3] =  cos_theta * cos_phi / sin_phi_safe
        LS[:, 1, 4] =  sin_theta * cos_phi / sin_phi_safe
        LS[:, 1, 5] = -1.0

        return LS

    # ------------------------------------------------------------------
    # Photometric interaction matrix (Eq. 29)
    # ------------------------------------------------------------------

    def interaction(self):
        """
        PS-VS photometric interaction matrix:
            L_IS = -∇I_S^T · L_S     (Eq. 29)

        where ∇I_S = (∂I_S/∂φ, ∂I_S/∂θ) and L_S is Eq. (30).

        Returns:
            LI: (N, 6)
        """
        # Spherical gradients via finite differences
        dI_dphi, dI_dtheta = self._compute_spherical_gradients()

        # Geometric interaction matrix
        LS = self._compute_LS()  # (N, 2, 6)

        # Stack gradients: (N, 2)
        grad_IS = torch.stack([dI_dphi, dI_dtheta], dim=1)

        # L_IS = -∇I_S^T · L_S
        LI = -torch.bmm(grad_IS.unsqueeze(1), LS).squeeze(1)  # (N, 6)

        return LI

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def error(self, s_star):
        assert self.I.shape == s_star.I.shape, \
            f"Dimension mismatch: {self.I.shape} vs {s_star.I.shape}"
        self._error = self.I - s_star.I
        return self._error

    # ------------------------------------------------------------------
    # Weighted interaction (M-estimators)
    # ------------------------------------------------------------------

    def weighted_interaction(self, s_star, estimator="tukey", param=4.685):
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
        self._error = None
