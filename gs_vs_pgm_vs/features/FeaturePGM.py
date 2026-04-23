"""
Photometric Gaussian Mixture (PGM) feature for visual servoing.

Based on: Crombez, Mouaddib, Caron, Chaumette,
"Visual Servoing with Photometric Gaussian Mixtures as Dense Feature",
IEEE Transactions on Robotics, 2019.

Implements the PGC (Photometric Gaussian Consistency) method for the
interaction matrix (Eq. 28) and the extension parameter gradient (Eq. 31).
"""

import torch
import torch.nn.functional as F
import math


class FeaturePGM:
    def __init__(self, lambda_g=10.0, border=10, device='cuda'):
        """
        Args:
            lambda_g: Gaussian extension parameter (sigma of the 2D Gaussian)
            border: number of border pixels to exclude
            device: torch device
        """
        self.lambda_g = lambda_g
        self.border = border
        self.device = torch.device(device)
        self.cam = None
        self.nbr = 0
        self.nbc = 0
        self.dim_s = 0

        # Storage
        self.G = None       # Gaussian mixture G(I, u_g, lambda_g)
        self.K_u = None     # sum_u I(u) dE/du_g
        self.K_v = None     # sum_u I(u) dE/dv_g
        self.Lambda = None  # dG/d(lambda_g) for extension parameter optimization
        self.Zs = None      # depth values (constant, from desired pose)
        self.xs = None      # normalized x coordinates of u_g
        self.ys = None      # normalized y coordinates of u_g
        self.s = None       # feature vector (flattened G)
        self._error = None

    def init(self, height, width):
        self.nbr = int(height)
        self.nbc = int(width)
        self.dim_s = (height - 2 * self.border) * (width - 2 * self.border)

    def setCameraParameters(self, cam):
        self.cam = cam

    def setLambda(self, lambda_g):
        self.lambda_g = lambda_g

    def _build_gaussian_kernels(self, lambda_g):
        """
        Build 1D Gaussian kernel and its derivative for separable convolution.

        Gaussian:       E(t) = exp(-t^2 / (2 * lambda_g^2))
        Derivative:     dE/dt_g = -(t_g - t)/lambda_g^2 * E = t/lambda_g^2 * E
                        (since kernel coord t = u_g - u, and dE/du_g uses chain rule)
        Lambda deriv:   dE/d(lambda_g) = t^2/lambda_g^3 * E

        Returns:
            k_gauss: 1D Gaussian kernel [K]
            k_deriv: 1D Gaussian first derivative kernel [K]  (for K_u, K_v)
            k_lambda: 1D kernel for t^2/lambda_g^3 * E [K]  (for Lambda computation)
        """
        # Kernel half-size: 3*sigma, minimum 1
        half = max(int(math.ceil(3.0 * lambda_g)), 1)
        size = 2 * half + 1

        t = torch.arange(-half, half + 1, dtype=torch.float32, device=self.device)
        lam2 = lambda_g * lambda_g

        gauss = torch.exp(-t * t / (2.0 * lam2))
        # Normalize so that convolution preserves intensity scale
        # (paper defines G as sum, not normalized convolution)
        # We do NOT normalize — the paper uses unnormalized sum (Eq. 4).

        # dE/du_g = -t / lam2 * E  analytically (where t = u_g - u).
        # F.conv2d computes cross-correlation (no kernel flip),
        # so it evaluates k(-t) instead of k(t). Since deriv is odd,
        # this implicitly flips the sign. To compensate, we use +t:
        # UPDATE: after testing, keeping -t here and adjusting the
        # interaction matrix sign is more consistent with the paper.
        deriv = (-t / lam2) * gauss

        # dE/d(lambda_g) = t^2/lambda_g^3 * E  (Eq. 30)
        lam_deriv = (t * t / (lambda_g ** 3)) * gauss

        return gauss, deriv, lam_deriv

    def _separable_conv(self, image, kernel_h, kernel_v):
        """
        Apply separable 2D convolution: first horizontal, then vertical.

        Args:
            image: [H, W] tensor
            kernel_h: 1D kernel for horizontal (column) direction
            kernel_v: 1D kernel for vertical (row) direction

        Returns:
            result: [H, W] tensor
        """
        kh = len(kernel_h)
        kv = len(kernel_v)
        ph = kh // 2
        pv = kv // 2

        img = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Horizontal pass
        kh_w = kernel_h.view(1, 1, 1, kh)
        img = F.pad(img, (ph, ph, 0, 0), mode='constant', value=0)
        img = F.conv2d(img, kh_w)

        # Vertical pass
        kv_w = kernel_v.view(1, 1, kv, 1)
        img = F.pad(img, (0, 0, pv, pv), mode='constant', value=0)
        img = F.conv2d(img, kv_w)

        return img[0, 0]

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build PGM features from grayscale image and depth.

        Computes:
        - G(I, u_g, lambda_g) = sum_u I(u) E(u_g - u)          (Eq. 4)
        - K_u = sum_u I(u) dE(u_g-u)/du_g                      (for Eq. 28)
        - K_v = sum_u I(u) dE(u_g-u)/dv_g                      (for Eq. 28)
        - Lambda_G = sum_u I(u) dE/d(lambda_g)                  (Eq. 31)

        Args:
            image: [H, W] grayscale image tensor (0-255 scale)
            depth: [H, W] depth map (Z-depth). If None, uses Z=1.
            mask: [H, W] boolean mask (True = valid). Optional.
        """
        assert isinstance(image, torch.Tensor)
        image = image.to(self.device).float()

        lam = self.lambda_g
        k_gauss, k_deriv, k_lam_deriv = self._build_gaussian_kernels(lam)

        # Zero out borders of the image for zero-border assumption (Eq. 20)
        img_padded = image.clone()
        img_padded[:self.border, :] = 0
        img_padded[-self.border:, :] = 0
        img_padded[:, :self.border] = 0
        img_padded[:, -self.border:] = 0

        # G(I, u_g, lambda_g) = I conv E  (separable: E_x * E_y)
        G_full = self._separable_conv(img_padded, k_gauss, k_gauss)

        # K_u = I conv (dE/du_g) = I conv (deriv_x * gauss_y)
        K_u_full = self._separable_conv(img_padded, k_deriv, k_gauss)

        # K_v = I conv (dE/dv_g) = I conv (gauss_x * deriv_y)
        K_v_full = self._separable_conv(img_padded, k_gauss, k_deriv)

        # Lambda gradient: sum of two separable terms
        # Term1: I conv ((t^2/lam^3 * E_x) * E_y)
        # Term2: I conv (E_x * (t^2/lam^3 * E_y))
        Lam1 = self._separable_conv(img_padded, k_lam_deriv, k_gauss)
        Lam2 = self._separable_conv(img_padded, k_gauss, k_lam_deriv)
        Lambda_full = Lam1 + Lam2

        # Crop to valid region (excluding borders)
        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        G_valid = G_full[i0:i1, j0:j1]
        K_u_valid = K_u_full[i0:i1, j0:j1]
        K_v_valid = K_v_full[i0:i1, j0:j1]
        Lambda_valid = Lambda_full[i0:i1, j0:j1]

        # Camera parameters
        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        # Normalized coordinates grid for u_g
        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device, dtype=torch.float32),
            torch.arange(i0, i1, device=self.device, dtype=torch.float32),
            indexing='xy'
        )
        x_g = (jj - u0) / px  # normalized x
        y_g = (ii - v0) / py  # normalized y

        # Depth handling
        if depth is not None:
            depth = depth.to(self.device).float()
            Z_valid = depth[i0:i1, j0:j1]
            Z_valid = torch.where(
                (Z_valid <= 1e-6) | torch.isnan(Z_valid) | torch.isinf(Z_valid),
                torch.tensor(1.0, device=self.device),
                Z_valid
            )
        else:
            Z_valid = torch.ones(i1 - i0, j1 - j0, device=self.device)

        # Flatten
        flat_G = G_valid.reshape(-1)
        flat_Ku = K_u_valid.reshape(-1)
        flat_Kv = K_v_valid.reshape(-1)
        flat_Lam = Lambda_valid.reshape(-1)
        flat_xg = x_g.reshape(-1)
        flat_yg = y_g.reshape(-1)
        flat_Z = Z_valid.reshape(-1)

        # Apply mask
        if mask is not None:
            mask = mask.to(self.device)
            mask_valid = mask[i0:i1, j0:j1].reshape(-1)
            flat_G = flat_G[mask_valid]
            flat_Ku = flat_Ku[mask_valid]
            flat_Kv = flat_Kv[mask_valid]
            flat_Lam = flat_Lam[mask_valid]
            flat_xg = flat_xg[mask_valid]
            flat_yg = flat_yg[mask_valid]
            flat_Z = flat_Z[mask_valid]
            self.dim_s = mask_valid.sum().item()

        self.G_2d = G_valid  # 2D for visualization
        self.G = flat_G
        self.K_u = flat_Ku
        self.K_v = flat_Kv
        self.Lambda = flat_Lam
        self.xs = flat_xg
        self.ys = flat_yg
        self.Zs = flat_Z
        self.s = flat_G.clone()

    def interaction(self):
        """
        PGC interaction matrix (Eq. 28 from Crombez et al. 2019).

        L_G = [L_G_vx  L_G_vy  L_G_vz  L_G_wx  L_G_wy  L_G_wz]

        where:
            L_G_vx = alpha_u * K_u / Z
            L_G_vy = alpha_v * K_v / Z
            L_G_vz = -(alpha_u * K_u * x_g + alpha_v * K_v * y_g) / Z
            L_G_wx = -alpha_u * K_u * x_g * y_g - alpha_v * K_v * (1 + y_g^2)
            L_G_wy = alpha_u * K_u * (1 + x_g^2) + alpha_v * K_v * x_g * y_g
            L_G_wz = -alpha_u * K_u * y_g + alpha_v * K_v * x_g

        Returns:
            LG: (N, 6) interaction matrix
        """
        au = self.cam.px  # alpha_u
        av = self.cam.py  # alpha_v

        Ku = self.K_u
        Kv = self.K_v
        xg = self.xs
        yg = self.ys
        Zinv = 1.0 / (self.Zs + 1e-8)

        N = Ku.shape[0]
        LG = torch.zeros((N, 6), device=self.device)

        LG[:, 0] = au * Ku * Zinv                                      # v_x
        LG[:, 1] = av * Kv * Zinv                                      # v_y
        LG[:, 2] = -(au * Ku * xg + av * Kv * yg) * Zinv               # v_z
        LG[:, 3] = -au * Ku * xg * yg - av * Kv * (1.0 + yg * yg)     # w_x
        LG[:, 4] = au * Ku * (1.0 + xg * xg) + av * Kv * xg * yg      # w_y
        LG[:, 5] = -au * Ku * yg + av * Kv * xg                        # w_z

        return LG

    def interaction_extended(self):
        """
        Extended interaction matrix including the lambda_g column (Eq. 33).

        L_G_lambda = [L_G(u_g),  Lambda_G(u_g)]

        Returns:
            LG_ext: (N, 7) extended interaction matrix
                    columns 0-5: camera velocity components
                    column 6: extension parameter derivative
        """
        LG = self.interaction()  # (N, 6)
        Lambda_col = self.Lambda.unsqueeze(1)  # (N, 1)
        return torch.cat([LG, Lambda_col], dim=1)

    def error(self, s_star):
        """
        PGM error: epsilon = G(I(r*), lambda_g*) - G(I(r), lambda_g)

        Note: sign is (desired - current) to match the PGM interaction matrix
        convention where v = -mu * L_G^+ * e produces descent.

        Args:
            s_star: desired PGM feature (FeaturePGM instance)

        Returns:
            error: (N,) error vector
        """
        self._error = s_star.s - self.s
        return self._error

    def reset(self):
        self._error = None
        self.G = None
        self.K_u = None
        self.K_v = None
        self.Lambda = None
