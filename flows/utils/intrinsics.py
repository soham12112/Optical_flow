"""Camera intrinsics utilities."""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CameraIntrinsics:
    """Camera intrinsics holder."""
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        """
        Args:
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point coordinates
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
    
    def scale(self, scale_x: float, scale_y: float) -> 'CameraIntrinsics':
        """
        Scale intrinsics for resized images.
        
        Args:
            scale_x: Width scaling factor
            scale_y: Height scaling factor
        
        Returns:
            New CameraIntrinsics object
        """
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )
    
    def __repr__(self):
        return f"CameraIntrinsics(fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f})"


def load_intrinsics(path: str) -> CameraIntrinsics:
    """
    Load camera intrinsics from JSON file.
    
    Expected format:
    {
        "fx": 500.0,
        "fy": 500.0,
        "cx": 320.0,
        "cy": 240.0
    }
    
    Args:
        path: Path to JSON file
    
    Returns:
        CameraIntrinsics object
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    return CameraIntrinsics(
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
    )


def save_intrinsics(intrinsics: CameraIntrinsics, path: str):
    """Save intrinsics to JSON file."""
    with open(path, 'w') as f:
        json.dump(intrinsics.to_dict(), f, indent=2)


def estimate_intrinsics(width: int, height: int, fov_degrees: float = 60.0) -> CameraIntrinsics:
    """
    Estimate intrinsics from image dimensions assuming a field of view.
    
    Args:
        width: Image width
        height: Image height
        fov_degrees: Horizontal field of view in degrees
    
    Returns:
        CameraIntrinsics object
    """
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov_degrees)
    
    # Compute focal length: f = (width / 2) / tan(fov / 2)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    
    # Assume square pixels
    fy = fx
    
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    
    logger.warning(
        f"Estimated intrinsics (assuming {fov_degrees}° FOV): "
        f"fx={fx:.1f}, fy={fy:.1f}. For accurate scene flow, provide actual intrinsics."
    )
    
    return CameraIntrinsics(fx, fy, cx, cy)


def backproject_depth(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Backproject depth map to 3D point cloud.
    
    Args:
        depth: Depth map (H, W) in arbitrary units
        intrinsics: Camera intrinsics
        mask: Optional binary mask (H, W), only backproject masked pixels
    
    Returns:
        Point cloud (H, W, 3) in camera coordinates [X, Y, Z]
    """
    h, w = depth.shape
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Backproject
    X = (u - intrinsics.cx) * depth / intrinsics.fx
    Y = (v - intrinsics.cy) * depth / intrinsics.fy
    Z = depth
    
    points = np.stack([X, Y, Z], axis=-1)
    
    # Apply mask if provided
    if mask is not None:
        points = points * mask[..., None]
    
    return points

