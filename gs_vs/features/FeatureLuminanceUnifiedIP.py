import torch
import torch.nn.functional as F


class FeatureLuminanceUnifiedIP:
    """
    Photometric visual feature for Image-Plane Visual Servoing (IP-VS)
    with Unified Central Model (UCM).

    Reference: Caron, Marchand, Mouaddib – Autonomous Robots, 2013
    Equations (13)-(15) for the geometric interaction matrix L_x.

    NOTE on depth:
        In the UCM, the depth appearing in L_x is ρ = ||X|| (Euclidean distance),
        NOT Z along the optical axis. GSplat's "expected depth" (ED) is the
        rendered Euclidean distance, so it can be used directly as ρ.

    Valid for fisheye cameras with xi >= 0.
    Invalid pixels (where the Jacobian is undefined) are discarded.
    """

    def __init__(self, device="cuda", border=0):
        self.device = device
        self.border = border
        self.nbr = None
        self.nbc = None
        self.dim_s = None

        self.cam = None
        self.xi = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, height, width):
        self.nbr = int(height)
        self.nbc = int(width)
        self.dim_s = (height - 2 * self.border) * (width - 2 * self.border)

    def setCameraParameters(self, cam, xi=1.635):
        self.cam = cam
        if xi is not None:
            self.xi = xi

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def compute_gradients_image_plane(self, image):
        """7-tap Gaussian derivative filter for image gradients."""
        coeffs = torch.tensor(
            [112.0, 913.0, 2047.0], dtype=torch.float32, device=self.device
        ) / 8418.0
        kernel = torch.zeros(7, dtype=torch.float32, device=self.device)
        kernel[0], kernel[1], kernel[2] = -coeffs[2], -coeffs[1], -coeffs[0]
        kernel[4], kernel[5], kernel[6] =  coeffs[0],  coeffs[1],  coeffs[2]
        kernel_x = kernel.view(1, 1, 1, 7)
        kernel_y = kernel.view(1, 1, 7, 1)
        image = image.unsqueeze(0).unsqueeze(0)
        image_pad_x = F.pad(image, (3, 3, 0, 0), mode='reflect')
        image_pad_y = F.pad(image, (0, 0, 3, 3), mode='reflect')
        Ix = F.conv2d(image_pad_x, kernel_x)[0, 0]
        Iy = F.conv2d(image_pad_y, kernel_y)[0, 0]
        return Ix, Iy

    # ------------------------------------------------------------------
    # Build features
    # ------------------------------------------------------------------

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build visual features from image and depth.

        Args:
            image: [H, W] grayscale image tensor
            depth: [H, W] depth map (ρ = Euclidean distance from GSplat ED)
            mask:  [H, W] boolean mask (True = valid)
        """
        assert isinstance(image, torch.Tensor)
        assert self.cam is not None
        assert self.xi is not None

        image = image.to(self.device).float()
        H, W = image.shape

        if self.nbr is None or self.nbc is None:
            self.init(H, W)

        # Depth (ρ in UCM)
        if depth is None:
            Z = torch.ones_like(image)
        else:
            Z = depth.to(self.device).float()
            Z = torch.where(
                (Z <= 1e-6) | torch.isnan(Z) | torch.isinf(Z),
                torch.tensor(1.0, device=self.device), Z
            )

        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        Ix, Iy = self.compute_gradients_image_plane(image)

        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        I_valid = image[i0:i1, j0:j1]
        Ix_valid = Ix[i0:i1, j0:j1] * px
        Iy_valid = Iy[i0:i1, j0:j1] * py
        Z_valid = Z[i0:i1, j0:j1]

        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device),
            torch.arange(i0, i1, device=self.device),
            indexing='xy'
        )
        x = (jj - u0) / px
        y = (ii - v0) / py

        # Flatten
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        I_flat = I_valid.reshape(-1)
        Ix_flat = Ix_valid.reshape(-1)
        Iy_flat = Iy_valid.reshape(-1)
        Z_flat = Z_valid.reshape(-1)

        # UCM validity: 1 + (1 - ξ²)(x² + y²) > 0
        r2 = x_flat**2 + y_flat**2
        arg = 1.0 + (1.0 - self.xi**2) * r2
        ucm_valid = arg > 0

        if mask is not None:
            mask_flat = mask.to(self.device).bool()[i0:i1, j0:j1].reshape(-1)
            valid = ucm_valid & mask_flat
        else:
            valid = ucm_valid

        self.x = x_flat[valid]
        self.y = y_flat[valid]
        self.Z = Z_flat[valid]   # ρ (Euclidean distance)
        self.I = I_flat[valid]
        self.Ix = Ix_flat[valid]
        self.Iy = Iy_flat[valid]

        self.N = self.x.shape[0]
        self.dim_s = self.N
        self.s = self.I.clone()

        assert torch.isfinite(self.Z).all(), "NaN/Inf in depth"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Error
    # ------------------------------------------------------------------

    def error(self, s_star):
        assert self.I.shape[0] == s_star.I.shape[0], \
            f"Dimension mismatch: {self.I.shape[0]} vs {s_star.I.shape[0]}"
        return self.I - s_star.I

    # ------------------------------------------------------------------
    # Geometric interaction matrix L_x — Eq. (14)-(15) of Caron 2013
    # ------------------------------------------------------------------

    def compute_Lx(self):
        """
        Geometric interaction matrix for UCM image-plane point.

        From Caron et al. 2013 Eq. (14)-(15):

        Translation block L1:
            [0,0] = -(1 + x²(1 - ξ(α+ξ)) + y²) / (ρ(α+ξ))
            [0,1] = ξxy / ρ
            [0,2] = αx / ρ
            [1,0] = ξxy / ρ
            [1,1] = -(1 + y²(1 - ξ(α+ξ)) + x²) / (ρ(α+ξ))
            [1,2] = αy / ρ

        Rotation block L2:
            [0,3] = xy
            [0,4] = -((1+x²)α - ξy²) / (α+ξ)
            [0,5] = y
            [1,3] = ((1+y²)α - ξx²) / (α+ξ)
            [1,4] = -xy
            [1,5] = -x

        where α = sqrt(1 + (1-ξ²)(x²+y²))  and  ρ = ||X||.

        Returns:
            Lx: (N, 2, 6)
        """
        x = self.x
        y = self.y
        xi = self.xi
        rho_inv = 1.0 / self.Z   # 1/ρ  (depth = Euclidean distance)

        r2 = x**2 + y**2
        alpha = torch.sqrt(torch.clamp(1.0 + (1.0 - xi**2) * r2, min=1e-8))
        denom = alpha + xi         # (α + ξ)

        Lx = torch.zeros((self.N, 2, 6), device=self.device)

        # ============================================================
        # Translation block — Eq. (14)
        # ============================================================
        #
        # IMPORTANT: The off-diagonal entries [0,1] and [1,0] are
        #   ξ·x·y / ρ
        # NOT  x·y·(1-ξ·denom) / (ρ·denom)  [this was the previous bug]
        #
        Lx[:, 0, 0] = -(1.0 + x**2 * (1.0 - xi * denom) + y**2) * rho_inv / denom
        Lx[:, 0, 1] = xi * x * y * rho_inv           # ← CORRECTED
        Lx[:, 0, 2] = alpha * x * rho_inv

        Lx[:, 1, 0] = xi * x * y * rho_inv           # ← CORRECTED (symmetric)
        Lx[:, 1, 1] = -(1.0 + y**2 * (1.0 - xi * denom) + x**2) * rho_inv / denom
        Lx[:, 1, 2] = alpha * y * rho_inv

        # ============================================================
        # Rotation block — Eq. (15)
        # ============================================================
        Lx[:, 0, 3] = x * y
        Lx[:, 0, 4] = -((1.0 + x**2) * alpha - xi * y**2) / denom
        Lx[:, 0, 5] = y

        Lx[:, 1, 3] = ((1.0 + y**2) * alpha - xi * x**2) / denom
        Lx[:, 1, 4] = -x * y
        Lx[:, 1, 5] = -x

        return Lx

    # ------------------------------------------------------------------
    # Photometric interaction matrix
    # ------------------------------------------------------------------

    def compute_LI(self):
        """LI = -∇I^T · Lx   (Eq. 19)"""
        Lx = self.compute_Lx()
        grad_I = torch.stack([self.Ix, self.Iy], dim=1)  # (N, 2)
        LI = -torch.bmm(grad_I.unsqueeze(1), Lx).squeeze(1)
        return LI

    def interaction(self):
        """Public API: returns (N, 6) photometric interaction matrix."""
        return self.compute_LI()
