import numpy as np
import cv2    
import matplotlib.pyplot as plt

import open3d as o3d
import torch
import imageio
import numpy as np
import os


def pixel2normalized(pixels, fx, fy, cx, cy):
    x = (pixels[:, 0] - cx) / fx
    y = (pixels[:, 1] - cy) / fy
    return np.vstack([x, y]).T

def normalized2pixel(norm_pts, fx, fy, cx, cy):
    u = norm_pts[:, 0] * fx + cx
    v = norm_pts[:, 1] * fy + cy
    return np.vstack([u, v]).T
    
    
def normalized_to_3D(s_normalized, Z):

    x_n, y_n = s_normalized[:, 0], s_normalized[:, 1]
    X = x_n * Z
    Y = y_n * Z
    Z = np.full_like(X, Z) if np.isscalar(Z) else Z
    return np.stack([X, Y, Z], axis=1)
    
    
def three_D_to_normalized(points_3D):
    """
    Projects 3D points in the camera frame to normalized image coordinates.

    Args:
        points_3D: (N, 3) array of 3D points in camera coordinates.

    Returns:
        s_normalized: (N, 2) array of normalized image coordinates.
    """
    X = points_3D[:, 0]
    Y = points_3D[:, 1]
    Z = points_3D[:, 2]
    
    # Avoid division by zero
    eps = 1e-8
    Z = np.where(Z == 0, eps, Z)

    x_n = X / Z
    y_n = Y / Z

    return np.stack([x_n, y_n], axis=1)
    
# Improved feature detection and matching
def detect_and_match_orb_old(gray_start, gray_desired, num_features=50):
    orb = cv2.ORB_create(nfeatures=num_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp_start, des_start = orb.detectAndCompute(gray_start, None)
    kp_desired, des_desired = orb.detectAndCompute(gray_desired, None)

    matches = bf.match(des_start, des_desired)
    matches = sorted(matches, key=lambda x: x.distance)[:num_features]

    pts_start = np.float32([kp_start[m.queryIdx].pt for m in matches])
    pts_desired = np.float32([kp_desired[m.trainIdx].pt for m in matches])

    return pts_start, pts_desired, matches, kp_start, kp_desired
    
def detect_and_match_orb(gray_start, gray_desired, num_features=200, distance_ratio=0.75):
    orb = cv2.ORB_create(nfeatures=num_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Use ratio test, so no crossCheck

    # Detect and compute keypoints and descriptors
    kp_start, des_start = orb.detectAndCompute(gray_start, None)
    kp_desired, des_desired = orb.detectAndCompute(gray_desired, None)

    # KNN match (get top 2 matches per feature)
    matches = bf.knnMatch(des_start, des_desired, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            good_matches.append(m)

    # Sort and keep top matches
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:num_features]

    # Extract matched points
    pts_start = np.float32([kp_start[m.queryIdx].pt for m in good_matches])
    pts_desired = np.float32([kp_desired[m.trainIdx].pt for m in good_matches])

    return pts_start, pts_desired, good_matches, kp_start, kp_desired
    
    
    
# Optional: visualize matches to verify
def visualize_matches(gray_start, gray_desired, kp_start, kp_desired, matches):
    matched_img = cv2.drawMatches(
        gray_start, kp_start,
        gray_desired, kp_desired,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title("ORB Feature Matches")
    plt.axis("off")
    plt.show()
    

def draw_matches(img, pts_current, pts_desired, figsize=(10, 10), radius=4):
    """
    Display correspondences between current and desired points using matplotlib.
    
    Args:
        img: Grayscale or color image (H, W) or (H, W, 3)
        pts_current: (N, 2) array of current points (e.g., tracked)
        pts_desired: (N, 2) array of desired points (e.g., original)
        figsize: size of the matplotlib figure
        radius: radius of the scatter points
    """
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    pts_current = np.array(pts_current)
    pts_desired = np.array(pts_desired)

    # Draw correspondences
    for p1, p2 in zip(pts_current, pts_desired):
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=1)  # green line
        plt.scatter(*p1, color='red', s=radius**2)   # red: current
        plt.scatter(*p2, color='blue', s=radius**2)  # blue: desired

    plt.title("Point correspondences before normalization")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
 
def image_difference(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """
    Compute image difference between two RGBA images, offsetting difference by 128 and clamping to [0, 255].
    Args:
        I1 (np.ndarray): First image, shape (H, W, 4), dtype=uint8.
        I2 (np.ndarray): Second image, same shape as I1.
    Returns:
        np.ndarray: The difference image, same shape and dtype.
    """
    if I1.shape != I2.shape:
        raise ValueError(f"Images must be of the same size, got {I1.shape} and {I2.shape}.")

    # Convert to int16 to avoid overflow during subtraction
    diff = I1.astype(np.int16) - I2.astype(np.int16) #+ 128

    # Clip the result to the valid range [0, 255] and convert back to uint8
    diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)
    
    return diff_clipped
    
    

def visualize_raw_depth(raw_depth):

    # Handle possible invalid depth values (e.g., inf or nan)
    valid_depth = np.where(np.isfinite(raw_depth), raw_depth, 0)
    if np.max(valid_depth) > np.min(valid_depth):
        depth_normalized = (valid_depth - np.min(valid_depth)) / (np.max(valid_depth) - np.min(valid_depth))
    else:
        depth_normalized = np.zeros_like(valid_depth)

    depth_8bit = (depth_normalized * 255).astype(np.uint8)
    
    return depth_8bit
    
    
def create_colored_point_cloud(depth, rgb_8bit, camera):
    H, W = depth.shape
    fx, fy, cx, cy = camera.px, camera.py, camera.u0, camera.v0  # Pinhole model
    rgb_8bit = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR)
    # Create meshgrid of pixel coordinates
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    x = (xs - cx) * depth / fx
    y = (ys - cy) * depth / fy
    z = depth

    # Stack and reshape
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb_8bit.reshape(-1, 3) / 255.0  # Normalize to [0,1]

    # Remove invalid depth points (optional)
    valid = z.flatten() > 0
    points = points[valid]
    colors = colors[valid]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
    
    


def save_rendered_images(rgb, gray, depth_raw, mask=None, out_dir="./logs", prefix="frame"):

    os.makedirs(out_dir, exist_ok=True)

    # --- RGB ---
    rgb_cpu = rgb.detach().cpu().numpy()  # [3, H, W]

    # Handle both [H,W,3] (gsplat) and [3,H,W] (Inria)
    if rgb_cpu.ndim == 3 and rgb_cpu.shape[0] == 3:
        # [3,H,W] -> [H,W,3]
        rgb_cpu = np.transpose(rgb_cpu, (1, 2, 0))
    elif rgb_cpu.ndim == 3 and rgb_cpu.shape[2] == 3:
        # already [H,W,3]
        pass
    else:
        raise ValueError(f"Unexpected RGB shape: {rgb_cpu.shape}")
    """
    # Scale if needed
    if rgb_cpu.max() <= 1.0:
        rgb_cpu = rgb_cpu * 255.0

    rgb_img = rgb_cpu.clip(0, 255).astype(np.uint8)
    """


    # Normalize per-image for visualization
    rgb_min = rgb_cpu.min()
    rgb_max = rgb_cpu.max()

    if rgb_max > rgb_min:
        rgb_vis = (rgb_cpu - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb_vis = np.zeros_like(rgb_cpu)

    rgb_img = (rgb_vis * 255).clip(0, 255).astype(np.uint8)



    # --- Grayscale ---
    gray_cpu = gray.detach().cpu().squeeze().numpy()  # [H, W]
    if gray_cpu.max() <= 1.0:  # ← only scale if in [0,1]
        gray_cpu *= 255.0

    gray_img = gray_cpu.clip(0, 255).astype(np.uint8)  # [H, W]

    # --- Depth (visualization) ---
    depth_cpu = depth_raw.detach().cpu().numpy()  # [H, W]
    depth_vis = np.where(np.isfinite(depth_cpu), depth_cpu, 0)

    depth_min = np.min(depth_vis[depth_vis > 0]) if np.any(depth_vis > 0) else 0
    depth_max = np.max(depth_vis)

    if depth_max > depth_min:
        depth_vis = ((depth_vis - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

    # --- Save images ---
    imageio.imwrite(os.path.join(out_dir, f"{prefix}_rgb.png"), rgb_img)
    imageio.imwrite(os.path.join(out_dir, f"{prefix}_gray.png"), gray_img)
    imageio.imwrite(os.path.join(out_dir, f"{prefix}_depth.png"), depth_vis)

    # --- Optional mask ---
    if mask is not None:
        mask_cpu = mask.detach().cpu().numpy().astype(np.uint8) * 255
        if mask_cpu.ndim == 1:
            H, W = depth_cpu.shape
            mask_cpu = mask_cpu.reshape(H, W)
        imageio.imwrite(os.path.join(out_dir, f"{prefix}_mask.png"), mask_cpu)

    #print(f"Saved images to {out_dir}/")



def depth_to_mesh(depth_tensor, cam):
    depth = depth_tensor.detach().cpu().numpy()  # [H, W]
    H, W = depth.shape

    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    Z = depth
    X = (i - cam.u0) * Z / cam.px
    Y = (j - cam.v0) * Z / cam.py

    vertices = np.stack((X, Y, Z), axis=-1)  # [H, W, 3]
    return vertices.reshape(-1, 3), H, W

def build_mesh_from_depth(depth_tensor, rgb_tensor, cam):
    verts, H, W = depth_to_mesh(depth_tensor, cam)
    faces = create_faces(H, W)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Add color if RGB is available
    if rgb_tensor is not None:
        rgb_np = rgb_tensor.detach().cpu().numpy()
        rgb_np = np.transpose(rgb_np, (1, 2, 0))  # HWC
        colors = rgb_np.reshape(-1, 3).clip(0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    return mesh  
    
        
def create_faces(H, W):
    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            i = y * W + x
            faces.append([i, i + 1, i + W])
            faces.append([i + 1, i + W + 1, i + W])
    return np.array(faces)
    
        
def load_image(filename):
    return cv2.imread(filename, cv2)
    
    
def center_and_normalize(I):
    """
    Centers and normalizes a NumPy array or Torch tensor.
    Mean → 0, Std → 1
    """
    eps = 1e-6

    if isinstance(I, np.ndarray):
        I = I.astype(np.float32)
        return (I - np.mean(I)) / (np.std(I) + eps)

    elif isinstance(I, torch.Tensor):
        I = I.float()
        return (I - I.mean()) / (I.std() + eps)

    else:
        raise TypeError(f"Unsupported type {type(I)}. Expected np.ndarray or torch.Tensor.")

def normalize_mad(I):
    """
    Centers and normalizes a NumPy array or Torch tensor.
    Mean → 0, Normalization by mean absolute deviation (MAD).
    """
    eps = 1e-6

    if isinstance(I, np.ndarray):
        I = I.astype(np.float32)
        I_prime = I - np.mean(I)
        abs_I_prime = np.abs(I_prime)
        return I_prime / (np.mean(abs_I_prime) + eps)

    elif isinstance(I, torch.Tensor):
        I = I.float()
        I_prime = I - I.mean()
        abs_I_prime = torch.abs(I_prime)
        return I_prime / (torch.mean(abs_I_prime) + eps)

    else:
        raise TypeError(f"Unsupported type {type(I)}. Expected np.ndarray or torch.Tensor.")



def match_lighting(I, I_star):
    a = torch.sum(I * I_star) / (torch.sum(I**2) + 1e-6)
    b = I_star.mean() - a * I.mean()
    return a * I + b

def equalize_histogram_torch(I):
    '''
    Histogram equalization for a single-channel torch image.
    Input: I in [0, 255] or [0, 1]
    Output: Equalized image in [0, 255]
    '''
    I = I.float()
    if I.max() <= 1.0:
        I = I * 255.0

    # Flatten image
    I_flat = I.flatten()

    # Compute histogram and cumulative distribution
    hist = torch.histc(I_flat, bins=256, min=0, max=255)
    cdf = hist.cumsum(0)
    cdf = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6)

    # Map each pixel intensity to equalized value
    idx = torch.bucketize(I_flat.long(), torch.arange(256, device=I.device), right=True)
    idx = torch.clamp(idx - 1, 0, 255)
    I_eq = cdf[idx].reshape_as(I)

    return I_eq
    
def equalize_histogram_numpy(I):
    I = I.astype(np.uint8)
    return cv2.equalizeHist(I)    


def clahe_equalize_torch(I, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE for single-channel torch image.
    Input: I in [0, 255] or [0, 1]
    Output: Equalized image in [0, 255] torch.Tensor
    """
    I = I.detach().cpu().float()
    if I.max() <= 1.0:
        I = I * 255.0

    I_np = I.numpy().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    I_eq_np = clahe.apply(I_np)
    I_eq = torch.from_numpy(I_eq_np).to(I.device).float()
    return I_eq
    
           
def clahe_equalize_numpy(I, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE for single-channel NumPy image.
    Input: I in [0, 255] or [0, 1]
    Output: Equalized image in [0, 255]
    """
    I = I.astype(np.float32)
    if I.max() <= 1.0:
        I = I * 255.0

    I = I.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    I_eq = clahe.apply(I)
    return I_eq    
    

def estimate_planar_pose_change(img1_path, img2_path, focal=800.0, debug=False):
    """
    Estimate camera rotation (deg) and translation (relative units)
    between two roughly planar views.

    Parameters
    ----------
    img1_path : str
        Path to the start image.
    img2_path : str
        Path to the target image.
    focal : float
        Approximate focal length in pixels.
    debug : bool
        If True, show matched features.

    Returns
    -------
    R_deg : ndarray (3,)
        Estimated rotation (roll, pitch, yaw) in degrees.
    T_vec : ndarray (3,)
        Estimated translation vector (x, y, z).
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    h, w = img1.shape

    # Detect and match features
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    # Extract keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography estimation failed — not enough inliers.")

    # Camera intrinsics
    K = np.array([[focal, 0, w / 2],
                  [0, focal, h / 2],
                  [0, 0, 1]], dtype=np.float64)

    # Decompose homography → multiple (R, T, n) solutions
    _, Rs, Ts, _ = cv2.decomposeHomographyMat(H, K)

    # Choose physically valid solution (positive Z translation)
    idx = np.argmax([T[2] for T in Ts])
    R, T = Rs[idx], Ts[idx]

    # Convert rotation to degrees
    rvec, _ = cv2.Rodrigues(R)
    R_deg = np.rad2deg(rvec.flatten())

    if debug:
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
        cv2.imshow("Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return R_deg, T.flatten()


import cv2
import numpy as np
import torch

def estimate_pose_from_torch_images(img1_torch, img2_torch, focal=800.0, debug=False):
    """
    Estimate rotation (in degrees) and translation (relative) between two planar views.
    Accepts torch tensors in [0,1], shape (1,H,W) or (H,W).

    Parameters
    ----------
    img1_torch, img2_torch : torch.Tensor
        Grayscale images (1,H,W) or (H,W) in [0,1].
    focal : float
        Approximate focal length in pixels.
    debug : bool
        If True, display ORB matches.

    Returns
    -------
    R_deg : np.ndarray
        Rotation vector in degrees (roll, pitch, yaw).
    T_vec : np.ndarray
        Translation vector (x, y, z) in relative units.
    """

    # Convert to numpy uint8
    def to_uint8(img_t):
        if img_t.ndim == 3:
            img_t = img_t.squeeze(0)
        img_np = (img_t.detach().cpu().numpy()).astype(np.uint8)
        return img_np

    img1 = to_uint8(img1_torch)
    img2 = to_uint8(img2_torch)
    h, w = img1.shape

    # ORB feature detection and matching
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("No features detected in one or both images.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    # Extract corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Homography estimation (planar assumption)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography estimation failed — not enough inliers.")

    # Camera intrinsics
    K = np.array([[focal, 0, w / 2],
                  [0, focal, h / 2],
                  [0, 0, 1]], dtype=np.float64)

    # Decompose homography into rotation + translation
    _, Rs, Ts, _ = cv2.decomposeHomographyMat(H, K)
    idx = np.argmax([T[2] for T in Ts])
    R, T = Rs[idx], Ts[idx]

    # Convert to rotation vector (Rodrigues)
    rvec, _ = cv2.Rodrigues(R)
    R_deg = np.rad2deg(rvec.flatten())

    if debug:
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
        cv2.imshow("Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return R_deg, T.flatten()    
    
def save_image_histogram(img, save_path="histogram.png", bins=256, title="Image Histogram"):
    """
    Compute and save the histogram of a grayscale image (PyTorch or NumPy).

    Parameters
    ----------
    img : torch.Tensor or np.ndarray
        Grayscale image tensor or array in [0,1] or [0,255].
        Shape can be (H, W) or (1, H, W).
    save_path : str
        Path to save the histogram image (e.g. 'histogram.png').
    bins : int
        Number of histogram bins.
    title : str
        Plot title.
    """

    # Convert to NumPy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().squeeze().numpy()

    # Flatten and scale if needed
    if img.max() <= 1.0:
        img = img * 255.0
    img = img.astype(np.uint8).ravel()

    # Compute and plot histogram
    plt.figure()
    plt.hist(img, bins=bins, range=(0, 255), color='gray')
    plt.title(title)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to: {save_path}")  
    
    
import torch
from skimage import io, exposure, img_as_ubyte
def match_histogram_torch(source, reference, nbins=256):
    """
    Match the histogram of a source image tensor to a reference image tensor.
    Both should be in range [0, 1].
    """
    # Flatten
    src = source.flatten()
    ref = reference.flatten()

    # Compute histograms
    src_hist = torch.histc(src, bins=nbins, min=0.0, max=1.0)
    ref_hist = torch.histc(ref, bins=nbins, min=0.0, max=1.0)

    # Compute CDFs
    src_cdf = torch.cumsum(src_hist, dim=0)
    src_cdf = src_cdf / src_cdf[-1]

    ref_cdf = torch.cumsum(ref_hist, dim=0)
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Create a mapping from source → reference
    interp_values = torch.linspace(0, 1, nbins)
    mapping = torch.zeros_like(interp_values)
    ref_idx = 0
    for i in range(nbins):
        while ref_idx < nbins - 1 and ref_cdf[ref_idx] < src_cdf[i]:
            ref_idx += 1
        mapping[i] = interp_values[ref_idx]

    # Map source pixels
    src_bin = torch.clamp((src * (nbins - 1)).long(), 0, nbins - 1)
    matched = mapping[src_bin].reshape_as(source)

    return matched
 
def compute_fisheye_mask(W, H, cx, cy, r_max, device="cuda"):
    """
    Create a circular mask for fisheye image
    
    Args:
        W, H: Image dimensions
        cx, cy: Principal point
        r_max: Maximum radius in pixels
        device: torch device
    
    Returns:
        mask: [H, W] bool tensor, True for valid pixels
    """
    u = torch.arange(W, dtype=torch.float32, device=device)
    v = torch.arange(H, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')
    
    # Distance from principal point
    r = torch.sqrt((uu - cx)**2 + (vv - cy)**2)
    
    # Valid if within circle
    mask = r <= r_max
    
    return mask
    
def compute_fisheye_mask_v2(width, height, cx, cy, device="cuda"):

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij"
    )

    radius = min(width, height) / 2.0

    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    return mask
