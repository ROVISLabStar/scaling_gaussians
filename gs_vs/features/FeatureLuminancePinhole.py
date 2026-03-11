import torch
import torch.nn.functional as F
import math

class FeatureLuminancePinhole:
    def __init__(self, border=10, device='cuda'):
       
        self.border = border
        self.device = torch.device(device)
        self.cam = None
        self.nbr = 0
        self.nbc = 0
        self.dim_s = 0

        # Camera parameters
        self.xi = None  # ξ parameter for unified model
        
        # Storage
        self.xs = None
        self.ys = None
        self.I = None
        self.Ix = None
        self.Iy = None
        self.Zs = None
        self.s = None
        self._error = None
        
        # Store the original image for interpolation
        self.image_original = None

    def init(self, height, width):
        """Initialize image dimensions"""
        self.nbr = int(height)
        self.nbc = int(width)
        self.dim_s = (height - 2 * self.border) * (width - 2 * self.border)

    def setCameraParameters(self, cam):

        self.cam = cam
       
    def compute_gradients_image_plane(self, image):
        """
        Compute image gradients using Gaussian derivatives 
        
        Args:
            image: [H, W] grayscale image tensor
            
        Returns:
            Ix, Iy: Image gradients
        """
        coeffs = torch.tensor([112.0, 913.0, 2047.0], dtype=torch.float32, device=self.device) / 8418.0

        kernel = torch.zeros(7, dtype=torch.float32, device=self.device)
        kernel[0], kernel[1], kernel[2] = -coeffs[2], -coeffs[1], -coeffs[0]
        kernel[4], kernel[5], kernel[6] =  coeffs[0],  coeffs[1],  coeffs[2]

        kernel_x = kernel.view(1, 1, 1, 7)
        kernel_y = kernel.view(1, 1, 7, 1)

        image = image.unsqueeze(0).unsqueeze(0)
        image_pad_x = F.pad(image, (3,3,0,0), mode='reflect')
        image_pad_y = F.pad(image, (0,0,3,3), mode='reflect')

        Ix = F.conv2d(image_pad_x, kernel_x)[0, 0]
        Iy = F.conv2d(image_pad_y, kernel_y)[0, 0]
        return Ix, Iy

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build visual features from image and depth
        
        Args:
            image: [H, W] grayscale image tensor
            depth: [H, W] depth map tensor (optional)
            mask: [H, W] boolean mask tensor (optional, True for valid pixels)
        """
        assert isinstance(image, torch.Tensor), "Input must be a torch.Tensor"
        image = image.to(self.device).float()
        self.image_original = image  # Store for interpolation
        
        # Ensure camera parameters are on device
        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        # Compute gradients on image plane (always needed for IP-VS)
        Ix, Iy = self.compute_gradients_image_plane(image)

        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border

        I_valid = image[i0:i1, j0:j1]
        Ix_valid = Ix[i0:i1, j0:j1] * px
        Iy_valid = Iy[i0:i1, j0:j1] * py

        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device),
            torch.arange(i0, i1, device=self.device),
            indexing='xy'
        )
        
        # Normalized image coordinates
        x = (jj - u0) / px
        y = (ii - v0) / py

        if depth is not None:
            depth = depth.to(self.device).float()
            Z_valid = depth[i0:i1, j0:j1]
            Z_valid = torch.where(
                (Z_valid <= 1e-6) | torch.isnan(Z_valid) | torch.isinf(Z_valid),
                torch.tensor(1.0, device=self.device),
                Z_valid
            )
        else:
            Z_valid = torch.full_like(I_valid, 1.0)

        # Flatten everything
        self.xs = x.reshape(-1)
        self.ys = y.reshape(-1)
        self.I = I_valid.reshape(-1)
        self.Ix = Ix_valid.reshape(-1)
        self.Iy = Iy_valid.reshape(-1)
        self.Zs = Z_valid.reshape(-1)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.to(self.device)
            mask_valid = mask[i0:i1, j0:j1].reshape(-1)
            self.xs = self.xs[mask_valid]
            self.ys = self.ys[mask_valid]
            self.I = self.I[mask_valid]
            self.Ix = self.Ix[mask_valid]
            self.Iy = self.Iy[mask_valid]
            self.Zs = self.Zs[mask_valid]
            self.dim_s = mask_valid.sum().item()
        
        self.s = self.I.clone()
        assert torch.isfinite(self.Zs).all(), "NaN/Inf in Zs"
        

    
    def interaction(self):
        """
        IP-VS: Image Plane Photometric Visual Servoing
        
        L_I(x) = -∇I^T L_x
        where ∇I = [∂I/∂x, ∂I/∂y] and L_x is the geometric interaction matrix
        
        Returns:
            LI: (N, 6) interaction matrix
        """

        Lx = self._compute_Lx_pinhole()
        
        # L_I = -∇I^T L_x
        grad_I = torch.stack([self.Ix, self.Iy], dim=1)  # (N, 2)
        assert torch.isfinite(grad_I).all(), "NaNs in grad_I"
        LI = -torch.bmm(grad_I.unsqueeze(1), Lx).squeeze(1)  # (N, 6)
        return LI


    def _compute_Lx_pinhole(self):
        """
        Geometric interaction matrix for standard pinhole camera
        Classic equation from [Espiau, Chaumette, Rives 1992]
        
        L_x = [-1/Z    0    x/Z   xy        -(1+x²)    y    ]
        L_y = [ 0    -1/Z   y/Z   (1+y²)    -xy       -x    ]
        
        Returns: 
            Lx: (N, 2, 6) interaction matrix
        """
        N = self.dim_s
        Zinv = 1.0 / (self.Zs + 1e-8)
        x, y = self.xs, self.ys

        Lx = torch.zeros((N, 2, 6), device=self.device)
        
        # First row: d x / d r
        Lx[:, 0, 0] = -Zinv
        Lx[:, 0, 1] = 0.0
        Lx[:, 0, 2] = x * Zinv
        Lx[:, 0, 3] = x * y
        Lx[:, 0, 4] = -(1 + x**2)
        Lx[:, 0, 5] = y

        # Second row: d y / d r
        Lx[:, 1, 0] = 0.0
        Lx[:, 1, 1] = -Zinv
        Lx[:, 1, 2] = y * Zinv
        Lx[:, 1, 3] = 1 + y**2
        Lx[:, 1, 4] = -x * y
        Lx[:, 1, 5] = -x

        return Lx

    

    def error(self, s_star):
        """
        Compute photometric error
        
        Args:
            s_star: Desired features
            
        Returns:
            error: (N,) photometric errors
        """
        self._error = self.s - s_star.s
        return self._error

    def reset(self):
        """Reset error and stored image"""
        self._error = None
        self.image_original = None

    def weighted_interaction(self, s_star, estimator="tukey", param=4.685):
        """
        Compute weighted interaction matrix using M-estimators for robust estimation
        
        Args:
            s_star: Desired features
            estimator: "tukey" or "huber"
            param: Tuning parameter (c in the M-estimator)
            
        Returns:
            L_weighted: Weighted interaction matrix
            error_weighted: Weighted errors
            weights: Computed weights
        """
        if self._error is None:
            self.error(s_star)
        error = self._error.view(-1)

        if estimator == "tukey":
            # Tukey biweight: w = (1 - (e/c)²)² for |e| < c, 0 otherwise
            e2 = error ** 2
            weights = torch.zeros_like(error)
            mask = e2 < param ** 2
            weights[mask] = ((1 - (e2[mask] / param ** 2)) ** 2)
            
        elif estimator == "huber":
            # Huber: w = 1 for |e| ≤ c, w = c/|e| for |e| > c
            abs_error = torch.abs(error)
            weights = torch.ones_like(error)
            weights[abs_error > param] = param / abs_error[abs_error > param]
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        L = self.interaction()
        L_weighted = weights.view(-1, 1) * L
        error_weighted = weights * error

        return L_weighted, error_weighted, weights
