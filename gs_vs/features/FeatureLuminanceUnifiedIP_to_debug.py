import torch
import torch.nn.functional as F


class FeatureLuminanceUnifiedIP:
    """
    Photometric visual feature for Image-Plane Visual Servoing (IP-VS)
    with Unified Central Model (UCM).

    Valid for fisheye cameras with xi > 1.
    Invalid pixels (where the Jacobian is undefined) are discarded.
    """

    def __init__(self, device="cuda", border=0):
        """
        Args:
            device: Device for computation ('cuda' or 'cpu')
            border: number of pixels to ignore on each side (default: 0)
        """
        self.device = device
        self.border = border
        self.nbr = None  # Will be set in init()
        self.nbc = None  # Will be set in init()
        self.dim_s = None  # Will be set in init()
        
        # Camera parameters - must be set via setCameraParameters before buildFrom
        self.cam = None
        self.xi = None

    # ------------------------------------------------------------------
    # Initialize image dimensions
    # ------------------------------------------------------------------

    def init(self, height, width):
        """
        Initialize image dimensions
        
        Args:
            height: Image height in pixels
            width: Image width in pixels
            
        Sets:
            nbr: number of rows
            nbc: number of columns
            dim_s: number of valid pixels after border cropping
        """
        self.nbr = int(height)
        self.nbc = int(width)
        self.dim_s = (height - 2 * self.border) * (width - 2 * self.border)

    # ------------------------------------------------------------------
    # Camera parameter setter
    # ------------------------------------------------------------------

    def setCameraParameters(self, cam, xi=1.635):
        """
        Set camera parameters and xi value
        
        Args:
            cam: Camera object with attributes px, py, u0, v0
            xi: ξ parameter for unified camera model (optional)
                - If provided, sets self.xi to this value
                - If None, keeps existing self.xi value (or None if never set)
                
        Note:
            For UCM fisheye cameras, typical xi values:
            - xi > 1: wide field of view fisheye (e.g., 1.7)
            - xi = 1: parabolic mirror
            - 0 < xi < 1: catadioptric systems
            - xi = 0: pinhole camera (standard perspective)
        """
        self.cam = cam
        if xi is not None:
            self.xi = xi

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def compute_gradients_image_plane(self, image):
        """
        Compute image gradients using Gaussian derivatives (order 7)
        Used for IP-VS representation
        
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
        image_pad_x = F.pad(image, (3, 3, 0, 0), mode='reflect')
        image_pad_y = F.pad(image, (0, 0, 3, 3), mode='reflect')
        Ix = F.conv2d(image_pad_x, kernel_x)[0, 0]
        Iy = F.conv2d(image_pad_y, kernel_y)[0, 0]
        return Ix, Iy

    # ------------------------------------------------------------------
    # 1. Build feature from current image
    # ------------------------------------------------------------------

    def buildFrom(self, image, depth=None, mask=None):
        """
        Build visual features from image and depth
        
        Args:
            image: [H, W] grayscale image tensor
            depth: [H, W] depth map tensor (optional)
            mask: [H, W] boolean mask tensor (optional, True for valid pixels)
        """
        assert isinstance(image, torch.Tensor), "Input must be a torch.Tensor"
        assert self.cam is not None, "Camera parameters must be set via setCameraParameters() before buildFrom()"
        assert self.xi is not None, "xi parameter must be set via setCameraParameters() before buildFrom()"
        
        image = image.to(self.device).float()

        H, W = image.shape
        
        # Initialize dimensions if not already done
        if self.nbr is None or self.nbc is None:
            self.init(H, W)

        # Handle depth
        if depth is None:
            Z = torch.ones_like(image)
        else:
            Z = depth.to(self.device).float()
            # Clean depth values
            Z = torch.where(
                (Z <= 1e-6) | torch.isnan(Z) | torch.isinf(Z),
                torch.tensor(1.0, device=self.device),
                Z
            )

        # Ensure camera parameters are on device
        px = torch.tensor(self.cam.px, device=self.device).float()
        py = torch.tensor(self.cam.py, device=self.device).float()
        u0 = torch.tensor(self.cam.u0, device=self.device).float()
        v0 = torch.tensor(self.cam.v0, device=self.device).float()

        # --------------------------------------------------------------
        # Image gradients using Gaussian derivatives
        # --------------------------------------------------------------
        Ix, Iy = self.compute_gradients_image_plane(image)
        
        # --------------------------------------------------------------
        # Apply border cropping if specified
        # --------------------------------------------------------------
        i0, i1 = self.border, self.nbr - self.border
        j0, j1 = self.border, self.nbc - self.border
        
        I_valid = image[i0:i1, j0:j1]
        Ix_valid = Ix[i0:i1, j0:j1] * px
        Iy_valid = Iy[i0:i1, j0:j1] * py
        Z_valid = Z[i0:i1, j0:j1]
        
        # Create coordinate grids
        jj, ii = torch.meshgrid(
            torch.arange(j0, j1, device=self.device),
            torch.arange(i0, i1, device=self.device),
            indexing='xy'
        )
        
        # Normalized image coordinates
        x = (jj - u0) / px
        y = (ii - v0) / py
        
        # Flatten everything
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        I_flat = I_valid.reshape(-1)
        Ix_flat = Ix_valid.reshape(-1)
        Iy_flat = Iy_valid.reshape(-1)
        Z_flat = Z_valid.reshape(-1)

        # --------------------------------------------------------------
        # UCM validity condition
        # --------------------------------------------------------------
        r2 = x_flat**2 + y_flat**2
        arg = 1.0 + (1.0 - self.xi**2) * r2
        
        ucm_valid = arg > 0
        
        # --------------------------------------------------------------
        # Combine UCM validity with user-provided mask
        # --------------------------------------------------------------
        if mask is not None:
            mask = mask.to(self.device).bool()
            mask_valid = mask[i0:i1, j0:j1].reshape(-1)
            # Combine both masks (UCM validity AND user mask)
            self.valid = ucm_valid & mask_valid
        else:
            self.valid = ucm_valid

        # Apply combined validity mask
        self.x = x_flat[self.valid]
        self.y = y_flat[self.valid]
        self.Z = Z_flat[self.valid]
        self.I = I_flat[self.valid]
        self.Ix = Ix_flat[self.valid]
        self.Iy = Iy_flat[self.valid]

        self.N = self.x.shape[0]
        self.dim_s = self.N  # Update actual number of valid pixels
        
        # Store feature vector (for compatibility)
        self.s = self.I.clone()
        
        # Verify no NaN/Inf in depth
        assert torch.isfinite(self.Z).all(), "NaN/Inf in depth values"

    # ------------------------------------------------------------------
    # Reset method
    # ------------------------------------------------------------------
    
    def reset(self):
        """Reset/clear current features (called between iterations)"""
        # Optional: clear feature data to free memory
        # For now, just pass since buildFrom will overwrite anyway
        pass

    # ------------------------------------------------------------------
    # 2. Photometric error
    # ------------------------------------------------------------------

    def error(self, s_star):
        """
        Photometric error vector e = I - I*
        
        Args:
            s_star: Desired feature object (another FeatureLuminanceUnifiedIP instance)
        
        Returns:
            error: (N,) tensor of photometric errors for valid pixels
        """
        # Ensure both features have same number of valid pixels
        assert self.I.shape[0] == s_star.I.shape[0], \
            f"Feature dimension mismatch: current has {self.I.shape[0]} pixels, desired has {s_star.I.shape[0]}"
        
        return self.I - s_star.I

    # ------------------------------------------------------------------
    # 3. Interaction matrix Lx (UCM, image plane)
    # ------------------------------------------------------------------

    def compute_Lx(self):
        """
        Compute geometric interaction matrix Lx (N, 2, 6)
        following Caron et al. (IP-VS, unified model).
        
        NOTE: At this point, self.x, self.y, self.Z are already filtered
        to contain only valid pixels where arg > 0, so sqrt is safe.
        """

        x = self.x
        y = self.y
        Zinv = 1.0 / self.Z
        xi = self.xi

        r2 = x**2 + y**2
        
        # Safe to compute alpha here because invalid pixels were filtered in buildFrom()
        arg = 1.0 + (1.0 - xi**2) * r2
        alpha = torch.sqrt(arg)  # Safe because arg > 0 for all remaining pixels
        
        denom = alpha + xi

        Lx = torch.zeros((self.N, 2, 6), device=self.device)

        # Translation components
        Lx[:, 0, 0] = -(1 + x**2 * (1 - xi * denom) + y**2) * Zinv / denom
        Lx[:, 0, 1] = x * y * (1 - xi * denom) * Zinv / denom
        Lx[:, 0, 2] = alpha * x * Zinv

        Lx[:, 1, 0] = x * y * (1 - xi * denom) * Zinv / denom
        Lx[:, 1, 1] = -(1 + y**2 * (1 - xi * denom) + x**2) * Zinv / denom
        Lx[:, 1, 2] = alpha * y * Zinv

        # Rotation components
        Lx[:, 0, 3] = x * y
        Lx[:, 0, 4] = -((1 + x**2) * alpha - xi * y**2) / denom
        Lx[:, 0, 5] = y

        Lx[:, 1, 3] = ((1 + y**2) * alpha - xi * x**2) / denom
        Lx[:, 1, 4] = -x * y
        Lx[:, 1, 5] = -x

        return Lx

    # ------------------------------------------------------------------
    # 4. Photometric interaction matrix LI
    # ------------------------------------------------------------------

    def compute_LI(self):
        """
        LI = - ∇I^T Lx
        """
        Lx = self.compute_Lx()
        grad_I = torch.stack([self.Ix, self.Iy], dim=1)  # (N, 2)

        LI = -torch.bmm(grad_I.unsqueeze(1), Lx).squeeze(1)
        return LI

    # ------------------------------------------------------------------
    # 5. Get interaction matrix (public API)
    # ------------------------------------------------------------------

    def interaction(self):
        """
        Compute and return the full photometric interaction matrix LI
        
        The interaction matrix relates the time derivative of the photometric
        feature to the camera velocity: ṡ = L_I * v_c
        
        Returns:
            LI: (N, 6) interaction matrix where:
                - N is the number of valid pixels
                - Columns represent [v_x, v_y, v_z, ω_x, ω_y, ω_z]
                - Each row is: -∇I^T * L_x for one pixel
        """
        return self.compute_LI()
