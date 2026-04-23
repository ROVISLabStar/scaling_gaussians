"""
Defocus-based Direct Visual Servoing (DDVS) Feature.

Based on: G. Caron, "Defocus-based Direct Visual Servoing",
IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 4057-4064, 2021.

Simulates camera defocus using the thin lens model:
  - Circle of Confusion (CoC) diameter: d(Z) = D*f/(Z_f-f) * (1 - Z_f/Z)
  - Gaussian spread: lambda(Z) = d(Z) / (6*k_u)
  - Defocus image: I_d(u) = I(u) * g(u, Z)  (depth-dependent Gaussian blur)

The interaction matrix includes the standard PVS terms plus a defocus term
for the Z-axis (Eq. 20 in the paper).

Author: Youssef ALJ (UM6P)
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


class FeatureDDVS:
    def __init__(self, focal_length=0.017, aperture_phi=8.0, focus_depth=0.25,
                 pixel_size=5.3e-6, border=10, device='cuda'):
        """
        Args:
            focal_length: f, lens focal length in meters (default: 17mm)
            aperture_phi: F-number phi (D = f/phi)
            focus_depth: Z_f, focus depth in meters
            pixel_size: k_u, physical size of a pixel in meters
            border: border pixels to exclude
            device: torch device
        """
        self.f = focal_length
        self.phi = aperture_phi
        self.D = focal_length / aperture_phi  # aperture diameter
        self.Z_f = focus_depth
        self.k_u = pixel_size
        self.border = border
        self.device = torch.device(device)
        self.cam = None
        self.nbr = 0
        self.nbc = 0

        # Storage
        self.s = None       # defocus image (flattened)
        self.I_d = None     # defocus image (2D)
        self.depth_map = None
        self.lambda_map = None  # per-pixel Gaussian spread
        self._error = None

    def init(self, height, width):
        self.nbr = int(height)
        self.nbc = int(width)

    def setCameraParameters(self, cam):
        self.cam = cam

    def _compute_coc_diameter(self, Z):
        """Circle of Confusion diameter d(Z) in meters (Eq. 9)."""
        return self.D * self.f / (self.Z_f - self.f) * (1.0 - self.Z_f / (Z + 1e-8))

    def _compute_lambda(self, Z):
        """Gaussian spread lambda(Z) in pixels (Eq. 11)."""
        d = self._compute_coc_diameter(Z)
        return torch.abs(d) / (6.0 * self.k_u)

    def _apply_defocus(self, image, depth):
        """
        Apply depth-dependent defocus blur to image.

        For efficiency, we discretize the depth into bins and apply
        a different Gaussian blur to each bin, then composite.
        """
        H, W = image.shape
        lam_map = self._compute_lambda(depth)  # per-pixel lambda in pixels
        self.lambda_map = lam_map

        # Clamp lambda to reasonable range
        lam_map = torch.clamp(lam_map, 0.0, 30.0)

        # Discretize into depth bins for efficiency
        n_bins = 10
        lam_min, lam_max = lam_map.min().item(), lam_map.max().item()
        if lam_max - lam_min < 0.1:
            # Nearly uniform defocus — apply single blur
            avg_lam = (lam_min + lam_max) / 2
            if avg_lam < 0.5:
                self.I_d = image.clone()
            else:
                self.I_d = self._gaussian_blur(image, avg_lam)
            return self.I_d

        bin_edges = torch.linspace(lam_min, lam_max + 1e-6, n_bins + 1, device=self.device)
        result = torch.zeros_like(image)
        weight = torch.zeros_like(image)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (lam_map >= lo) & (lam_map < hi)
            if not mask.any():
                continue

            avg_lam = ((lo + hi) / 2).item()
            if avg_lam < 0.3:
                blurred = image
            else:
                blurred = self._gaussian_blur(image, avg_lam)

            result += blurred * mask.float()
            weight += mask.float()

        weight = torch.clamp(weight, min=1.0)
        self.I_d = result / weight
        return self.I_d

    def _gaussian_blur(self, image, sigma):
        """Apply isotropic Gaussian blur with given sigma."""
        if sigma < 0.3:
            return image
        half = max(int(math.ceil(3.0 * sigma)), 1)
        size = 2 * half + 1
        t = torch.arange(-half, half + 1, dtype=torch.float32, device=self.device)
        kernel_1d = torch.exp(-t * t / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()

        img = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        # Horizontal
        kh = kernel_1d.view(1, 1, 1, size)
        img = F.pad(img, (half, half, 0, 0), mode='reflect')
        img = F.conv2d(img, kh)
        # Vertical
        kv = kernel_1d.view(1, 1, size, 1)
        img = F.pad(img, (0, 0, half, half), mode='reflect')
        img = F.conv2d(img, kv)
        return img[0, 0]

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build DDVS feature from grayscale image and depth.

        Args:
            image: (H, W) grayscale tensor
            depth: (H, W) depth map (Z values)
            mask: optional boolean mask
        """
        image = image.to(self.device).float()
        if depth is not None:
            depth = depth.to(self.device).float()
            depth = torch.where(depth < 1e-3, torch.tensor(1.0, device=self.device), depth)
        else:
            depth = torch.ones_like(image)

        self.depth_map = depth

        # Apply defocus blur
        I_d = self._apply_defocus(image, depth)

        # Crop to valid region
        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border
        I_d_valid = I_d[i0:i1, j0:j1]

        self.s = I_d_valid.reshape(-1)

    def error(self, s_star):
        """Compute error: I_d - I_d*"""
        self._error = self.s - s_star.s
        return self._error

    def interaction(self):
        """
        DDVS interaction matrix (Eq. 20).

        L_{I_d}(u) = [-nabla_u I_d^T] [L_u            ]
                     [-Delta_u I_d  ] [Df/(6k_u(Z_f-f)Z) L_Z]

        where L_u is the standard geometric interaction matrix (pinhole)
        and L_Z = [0, -1, -Y, X, 0] is the Z-motion geometric matrix.

        For simplicity, we use the standard PVS interaction matrix
        (which is a good approximation when defocus variation is small)
        plus the defocus correction term.
        """
        px = self.cam.px
        py = self.cam.py
        u0 = self.cam.u0
        v0 = self.cam.v0

        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        # Compute image gradients of defocus image
        I_d = self.I_d
        # 7-tap Gaussian derivative (same as pinhole feature)
        coeffs = torch.tensor([112.0, 913.0, 2047.0], device=self.device) / 8418.0
        kernel = torch.tensor([-coeffs[2], -coeffs[1], -coeffs[0], 0,
                                coeffs[0], coeffs[1], coeffs[2]], device=self.device)

        img = I_d.unsqueeze(0).unsqueeze(0)
        kh = kernel.view(1, 1, 1, 7)
        kv = kernel.view(1, 1, 7, 1)

        # Horizontal gradient
        Iu = F.conv2d(F.pad(img, (3, 3, 0, 0), mode='reflect'), kh)[0, 0]
        # Vertical gradient
        Iv = F.conv2d(F.pad(img, (0, 0, 3, 3), mode='reflect'), kv)[0, 0]

        # Crop
        Iu_valid = Iu[i0:i1, j0:j1].reshape(-1)
        Iv_valid = Iv[i0:i1, j0:j1].reshape(-1)

        # Depth
        Z_valid = self.depth_map[i0:i1, j0:j1].reshape(-1)
        Z_valid = torch.clamp(Z_valid, min=1e-3)

        # Normalized coordinates
        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device, dtype=torch.float32),
            torch.arange(i0, i1, device=self.device, dtype=torch.float32),
            indexing='xy')
        x = ((jj - u0) / px).reshape(-1)
        y = ((ii - v0) / py).reshape(-1)
        Zinv = 1.0 / Z_valid

        N = x.shape[0]
        Ls = torch.zeros((N, 6), device=self.device)

        # Standard PVS interaction matrix: LI = -[Ix, Iy] @ Lx
        # Ix = Iu*px, Iy = Iv*py (normalized gradients)
        Ix = Iu_valid * px
        Iy = Iv_valid * py

        Ls[:, 0] = Ix * Zinv                                    # v_x
        Ls[:, 1] = Iy * Zinv                                    # v_y
        Ls[:, 2] = -(Ix * x + Iy * y) * Zinv                    # v_z
        Ls[:, 3] = -(Ix * x * y + Iy * (1 + y * y))             # w_x
        Ls[:, 4] = Ix * (1 + x * x) + Iy * x * y               # w_y
        Ls[:, 5] = -Ix * y + Iy * x                             # w_z

        # DDVS defocus correction for v_z (Eq. 20)
        # Additional term: -Delta_u(I_d) * D*f / (6*k_u*(Z_f-f)*Z) * L_Z
        # L_Z for v_z component = -X/Z (from Eq. 21: L_Z = [0, -1, -Y, X, 0])
        # The Laplacian term adds to the v_z column
        laplacian = F.conv2d(
            F.pad(img, (1, 1, 1, 1), mode='reflect'),
            torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                          dtype=torch.float32, device=self.device)
        )[0, 0]
        lap_valid = laplacian[i0:i1, j0:j1].reshape(-1)

        defocus_coeff = self.D * self.f / (6.0 * self.k_u * (self.Z_f - self.f))
        # Add defocus contribution to v_z
        Ls[:, 2] += -lap_valid * defocus_coeff * Zinv

        return Ls

    def reset(self):
        self.s = None
        self.I_d = None
        self._error = None
