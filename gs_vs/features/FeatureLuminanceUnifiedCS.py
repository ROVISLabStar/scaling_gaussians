import torch
import torch.nn.functional as F


class FeatureLuminanceUnifiedCS:
    """
    Photometric visual feature for Cartesian Spherical Visual Servoing (CS-VS)
    using the Unified Central Camera Model.
    
    This version EXACTLY matches the CS-VS implementation from FeatureLuminanceTorchComplete
    to ensure identical behavior.

    Reference:
    Caron, Marchand, Mouaddib – Autonomous Robots, 2013 (Section 3.2)
    """

    def __init__(self, device="cuda", border=0):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.border = border

        self.nbr = 0
        self.nbc = 0
        self.dim_s = 0

        # Camera parameters
        self.cam = None
        self.xi = None

        # Stored features (matching TorchComplete structure)
        self.xs = None
        self.ys = None
        self.I = None
        self.Ix = None  # Not used in CS but kept for compatibility
        self.Iy = None  # Not used in CS but kept for compatibility
        self.Zs = None  # Depth - IMPORTANT for CS
        self.s = None
        
        # Needed for spherical gradients
        self.image_original = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, height, width):
        self.nbr = int(height)
        self.nbc = int(width)
        self.dim_s = (height - 2 * self.border) * (width - 2 * self.border)

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
    # Image gradients (kept for compatibility but not used in CS-VS)
    # ------------------------------------------------------------------

    def compute_gradients_image_plane(self, image):
        """Compute image gradients (not used in CS-VS but kept for compatibility)"""
        coeffs = torch.tensor(
            [112.0, 913.0, 2047.0],
            dtype=torch.float32,
            device=self.device
        ) / 8418.0

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

    # ------------------------------------------------------------------
    # Build feature 
    # ------------------------------------------------------------------

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build visual features from image and depth
        EXACTLY matches TorchComplete.buildFrom() behavior
        
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

        # Compute gradients on image plane (for compatibility)
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

        # Handle depth
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

        # Flatten everything (lines 168-174)
        self.xs = x.reshape(-1)
        self.ys = y.reshape(-1)
        self.I = I_valid.reshape(-1)
        self.Ix = Ix_valid.reshape(-1)
        self.Iy = Iy_valid.reshape(-1)
        self.Zs = Z_valid.reshape(-1)  # CRITICAL: Store depth
        
        # Apply mask if provided (lines 177-186)
        if mask is not None:
            mask = mask.to(self.device)
            mask_valid = mask[i0:i1, j0:j1].reshape(-1)
            self.xs = self.xs[mask_valid]
            self.ys = self.ys[mask_valid]
            self.I = self.I[mask_valid]
            self.Ix = self.Ix[mask_valid]
            self.Iy = self.Iy[mask_valid]
            self.Zs = self.Zs[mask_valid]  # CRITICAL: Also mask depth
            self.dim_s = mask_valid.sum().item()  # CRITICAL: Update dim_s
        
        self.s = self.I.clone()
        assert torch.isfinite(self.Zs).all(), "NaN/Inf in Zs"

    # ------------------------------------------------------------------
    # CS-VS Interaction (EXACTLY matching TorchComplete)
    # ------------------------------------------------------------------

    def interaction(self):
        """
        Compute CS-VS interaction matrix
        EXACTLY matches TorchComplete._interaction_CS()
        
        Returns:
            LI: (N, 6) interaction matrix
        """
        # 1. Compute spherical Cartesian gradients (line 437)
        dI_dXS = self.compute_spherical_gradients_finite_difference()  # (N, 3)
        
        # 2. Geometric interaction matrix for Cartesian spherical (line 440)
        LXS = self.interaction_cartesian_spherical()  # (N, 3, 6)
        
        # 3. L_I = -∇I_S^T L_XS (line 443)
        LI = -torch.einsum('ni,nij->nj', dI_dXS, LXS)  # (N, 6)
        
        return LI

    def interaction_cartesian_spherical(self):
        """
        Interaction matrix for Cartesian spherical representation
        EXACTLY matches TorchComplete.interaction_cartesian_spherical() (lines 447-470)
        
        L_XS = [ (X_S X_S^T - I₃)/ρ,  [X_S]× ]
        
        Returns: 
            LXS: (N, 3, 6) interaction matrix
        """
        XS = self.backproject_to_sphere_unified()  # (N, 3) - line 456
        # ρ = Euclidean distance ||X|| — Eq. (24) of Caron et al. 2013
        # GSplat "expected depth" (ED) is Euclidean distance, so Zs = ρ directly.
        rho = self.Zs.view(-1, 1)  # (N, 1)
        N = XS.shape[0]  # line 459
        LXS = torch.zeros((N, 3, 6), device=self.device)  # line 460
        
        # Translation block: (X_S X_S^T - I)/ρ (lines 462-465)
        XXT = torch.bmm(XS.unsqueeze(2), XS.unsqueeze(1))  # (N, 3, 3)
        I3 = torch.eye(3, device=self.device).unsqueeze(0).expand(N, -1, -1)
        LXS[:, :, :3] = (XXT - I3) / (rho.unsqueeze(-1) + 1e-8)

        # Rotation block: [X_S]× (skew-symmetric) (line 468)
        LXS[:, :, 3:] = self._skew_symmetric(XS)
        
        return LXS

    # ------------------------------------------------------------------
    # Spherical gradients
    # ------------------------------------------------------------------

    def compute_spherical_gradients_finite_difference(self, N=2):
        """
        Compute Cartesian spherical gradients ∇I_S = [∂I/∂X_S, ∂I/∂Y_S, ∂I/∂Z_S]
        EXACTLY matches TorchComplete.compute_spherical_gradients_finite_difference()
        (lines 840-901)
        
        Args:
            N: Order of finite differences (2 for central difference)
            
        Returns:
            dI_dXS: (N_points, 3) gradients on sphere
        """
        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()
        
        # Compute delta (equation 25) - lines 856-863
        x_plus1 = torch.tensor([1.0], device=self.device) / px
        y_plus1 = torch.tensor([0.0], device=self.device) / py
        
        XS_plus1 = self.backproject_to_sphere_coords(x_plus1, y_plus1)
        XS_center = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        
        delta = torch.norm(XS_plus1 - XS_center)
        
        XS = self.backproject_to_sphere_unified()
        n_points = XS.shape[0]
        
        dI_dXS = torch.zeros((n_points, 3), device=self.device)
        
        # Generate k values (lines 870-873)
        k_values = []
        for k in range(-N//2, N//2 + 1):
            if k != 0:
                k_values.append(k)
        
        # Get finite difference coefficients 
        coefficients = self._get_finite_difference_coefficients(k_values)
        
        # Compute gradients for each coordinate
        for coord_idx in range(3):
            intensities = {k: None for k in k_values}
            
            for k in k_values:
                XS_modified = XS.clone()
                XS_modified[:, coord_idx] += k * delta
                
                XS_norm = torch.norm(XS_modified, dim=1, keepdim=True)
                XS_neighbor = XS_modified / (XS_norm + 1e-8)
                
                x_norm, y_norm = self.sphere_to_image_plane_unified(XS_neighbor)
                
                u_pixel = x_norm * px + u0
                v_pixel = y_norm * py + v0
                
                I_k = self.interpolate_image(u_pixel, v_pixel)
                intensities[k] = I_k
            
            gradient = torch.zeros(n_points, device=self.device)
            for k, coeff in zip(k_values, coefficients):
                gradient += coeff * intensities[k]
            
            dI_dXS[:, coord_idx] = gradient / delta
        
        return dI_dXS

    def _get_finite_difference_coefficients(self, k_values):
        """
        Compute finite difference coefficients on GPU
        EXACTLY matches TorchComplete._get_finite_difference_coefficients() (lines 903-924)
        
        Args:
            k_values: List of offsets (e.g., [-1, 1] for central difference)
            
        Returns:
            coeffs: Finite difference coefficients
        """
        k_tensor = torch.tensor(k_values, dtype=torch.float32, device=self.device)
        n = len(k_values)
        
        powers = torch.arange(n, device=self.device).float().view(1, n)
        k_powers = k_tensor.view(-1, 1) ** powers
        
        rhs = torch.zeros(n, device=self.device)
        rhs[1] = 1.0
        
        coeffs = torch.linalg.solve(k_powers.T, rhs)
        
        return coeffs

    
    def backproject_to_sphere_unified(self):
        """
        Backproject using unified omnidirectional model
        EXACTLY matches TorchComplete.backproject_to_sphere_unified() (lines 711-729)
        
        Returns: 
            XS: (N, 3) points on unit sphere
        """
        x, y = self.xs, self.ys
        xi = self.xi

        r2 = x**2 + y**2
        eps = 1e-8

        alpha = torch.sqrt(torch.clamp(1 + (1 - xi**2) * r2, min=eps))
        denom = r2 + 1
        
        XS = torch.stack([
            (xi + alpha) * x / denom,
            (xi + alpha) * y / denom,
            (xi + alpha) / denom - xi
        ], dim=1)
        
        # Normalize to unit sphere
        XS_norm = torch.norm(XS, dim=1, keepdim=True)
        XS = XS / (XS_norm + eps)
        
        return XS

    def backproject_to_sphere_coords(self, x, y):
        """
        Backproject single normalized coordinates to sphere
        For unified model
        
        Args:
            x, y: Normalized image coordinates
            
        Returns:
            XS: (N, 3) points on sphere
        """
        xi = self.xi
        r2 = x**2 + y**2
        eps = 1e-8
        
        alpha = torch.sqrt(torch.clamp(1 + (1 - xi**2) * r2, min=eps))
        denom = r2 + 1
        
        XS = torch.stack([
            (xi + alpha) * x / denom,
            (xi + alpha) * y / denom,
            (xi + alpha) / denom - xi
        ], dim=1)
        
        XS_norm = torch.norm(XS, dim=1, keepdim=True)
        XS = XS / (XS_norm + eps)
        
        return XS
    

    
    def sphere_to_image_plane_unified(self, XS):
        """
        Unified model: x = X_S/(Z_S + ξ), y = Y_S/(Z_S + ξ)
        EXACTLY matches TorchComplete.sphere_to_image_plane_unified() (lines 780-792)
        
        Args:
            XS: (N, 3) points on sphere
            
        Returns:
            x, y: Normalized image coordinates
        """
        eps = 1e-8
        x = XS[:, 0] / (XS[:, 2] + self.xi + eps)
        y = XS[:, 1] / (XS[:, 2] + self.xi + eps)
        return x, y

    def interpolate_image(self, u, v):
        """
        Bilinear interpolation of image at coordinates (u, v)
        EXACTLY matches TorchComplete.interpolate_image() (lines 814-838)
        
        Args:
            u, v: Image coordinates (can be fractional)
            
        Returns:
            I_interp: Interpolated intensities
        """
        if self.image_original is None:
            raise ValueError("No image stored for interpolation")
        
        H, W = self.image_original.shape
        
        u_float = u.float()
        v_float = v.float()
        
        u0 = torch.floor(u_float).long()
        u1 = u0 + 1
        v0 = torch.floor(v_float).long()
        v1 = v0 + 1
        
        u0 = torch.clamp(u0, 0, W - 1)
        u1 = torch.clamp(u1, 0, W - 1)
        v0 = torch.clamp(v0, 0, H - 1)
        v1 = torch.clamp(v1, 0, H - 1)
        
        I00 = self.image_original[v0, u0]
        I01 = self.image_original[v0, u1]
        I10 = self.image_original[v1, u0]
        I11 = self.image_original[v1, u1]
        
        w00 = (u1.float() - u_float) * (v1.float() - v_float)
        w01 = (u_float - u0.float()) * (v1.float() - v_float)
        w10 = (u1.float() - u_float) * (v_float - v0.float())
        w11 = (u_float - u0.float()) * (v_float - v0.float())
        
        I_interp = w00 * I00 + w01 * I01 + w10 * I10 + w11 * I11
        
        return I_interp

    def _skew_symmetric(self, v):
        """
        Create skew-symmetric matrix from vector [X_S]×
        EXACTLY matches TorchComplete._skew_symmetric() (lines 1019-1032)
        
        Args:
            v: (N, 3) vectors
            
        Returns:
            S: (N, 3, 3) skew-symmetric matrices
        """
        N = v.shape[0]
        S = torch.zeros((N, 3, 3), device=self.device)
        S[:, 0, 1] = -v[:, 2]
        S[:, 0, 2] =  v[:, 1]
        S[:, 1, 0] =  v[:, 2]
        S[:, 1, 2] = -v[:, 0]
        S[:, 2, 0] = -v[:, 1]
        S[:, 2, 1] =  v[:, 0]
        return S

    # ------------------------------------------------------------------
    # Error and reset
    # ------------------------------------------------------------------

    def error(self, s_star):
        """
        Compute photometric error
        
        Args:
            s_star: Desired features
            
        Returns:
            error: (N,) photometric errors
        """
        return self.s - s_star.s

    def reset(self):
        """Reset error and stored image"""
        self.image_original = None
