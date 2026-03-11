import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from simulators.CameraParameters import CameraParameters


# ---- Image Simulator ----
class ImageSimulator:
    def __init__(self):
        self.texture = None
        self.plane_points = None
        self.cam_pose = np.eye(4)

    def init(self, texture, X):
        self.texture = texture
        self.plane_points = np.array(X)  # 4x3
        h, w = texture.shape[:2]
        self.tex_coords = np.array([[0, 0],
                                    [w - 1, 0],
                                    [w - 1, h - 1],
                                    [0, h - 1]], dtype=np.float32)

    def setCameraPosition(self, cMo):
        self.cam_pose = np.array(cMo)

    def getImage(self, output_shape, cam_params):
        H, W = output_shape
        projected = []
        for X in self.plane_points:
            X_cam = self.cam_pose[:3, :3] @ X + self.cam_pose[:3, 3]
            x = cam_params.px * (X_cam[0] / X_cam[2]) + cam_params.u0
            y = cam_params.py * (X_cam[1] / X_cam[2]) + cam_params.v0
            projected.append([x, y])
        projected = np.array(projected, dtype=np.float32)
        Hmat, _ = cv2.findHomography(self.tex_coords, projected)
        image = cv2.warpPerspective(self.texture, Hmat, (W, H), flags=cv2.INTER_LINEAR)
        return image
