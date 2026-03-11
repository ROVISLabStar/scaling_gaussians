import open3d as o3d

def save_colored_point_cloud(points_3d, rgb_init, mask=None, filename="colored_cloud.ply"):
    """
    Save a colored point cloud to a PLY file.

    Args:
        points_3d: Tensor [H, W, 3], 3D coordinates in camera/world frame
        rgb_init: Tensor [3, H, W], RGB image (range [0,1] or [0,255])
        mask: Optional binary mask [H, W] to filter valid points
        filename: Output filename (.ply)
    """
    H, W, _ = points_3d.shape
    colors = rgb_init.permute(1, 2, 0).detach().cpu().numpy().reshape(-1, 3)
    points = points_3d.detach().cpu().numpy().reshape(-1, 3)

    if colors.max() > 1.0:
        colors = colors / 255.0  # Normalize if needed

    if mask is not None:
        mask_flat = mask.flatten()
        points = points[mask_flat]
        colors = colors[mask_flat]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved {len(points)} points to {filename}")

