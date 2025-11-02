"""
Scene flow estimation using depth + pose fusion.
Implements Track 1: Depth + Pose + Flow -> 3D motion
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def estimate_pose_from_flow(
    flow: np.ndarray,
    intrinsics,
    weight: Optional[np.ndarray] = None,
    min_matches: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera pose (R, t) from optical flow using essential matrix.
    
    Args:
        flow: Optical flow (H, W, 2)
        intrinsics: CameraIntrinsics object
        weight: Optional weight map for selecting reliable correspondences
        min_matches: Minimum number of matches required
    
    Returns:
        Tuple of (R, t):
        - R: 3x3 rotation matrix
        - t: 3x1 translation vector (unit norm)
    """
    h, w = flow.shape[:2]
    
    # Create pixel coordinates
    y, x = np.mgrid[0:h, 0:w]
    pts1 = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)
    
    # Compute corresponding points
    flow_flat = flow.reshape(-1, 2)
    pts2 = pts1 + flow_flat
    
    # Filter by weight if provided
    if weight is not None:
        weight_flat = weight.flatten()
        # Select top weighted points
        threshold = np.percentile(weight_flat, 50)  # Use top 50%
        valid_mask = weight_flat > threshold
    else:
        # Use all points with sufficient flow magnitude
        flow_mag = np.linalg.norm(flow_flat, axis=1)
        valid_mask = flow_mag > 0.5
    
    pts1_valid = pts1[valid_mask]
    pts2_valid = pts2[valid_mask]
    
    if len(pts1_valid) < min_matches:
        logger.warning(f"Insufficient matches ({len(pts1_valid)} < {min_matches}), using identity pose")
        return np.eye(3), np.zeros((3, 1))
    
    # Sample for speed (use at most 5000 points)
    if len(pts1_valid) > 5000:
        indices = np.random.choice(len(pts1_valid), 5000, replace=False)
        pts1_valid = pts1_valid[indices]
        pts2_valid = pts2_valid[indices]
    
    # Compute essential matrix
    K = intrinsics.to_matrix()
    E, mask = cv2.findEssentialMat(
        pts1_valid,
        pts2_valid,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    
    if E is None:
        logger.warning("Essential matrix estimation failed, using identity pose")
        return np.eye(3), np.zeros((3, 1))
    
    # Recover pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_valid, pts2_valid, K, mask=mask)
    
    inliers = mask_pose.sum() if mask_pose is not None else 0
    logger.debug(f"Pose estimation: {inliers} inliers out of {len(pts1_valid)} points")
    
    return R, t


def compute_scene_flow_from_depth_pose(
    depth1: np.ndarray,
    depth2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    intrinsics,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute scene flow from depth maps and camera pose.
    
    Scene flow S(x) = P'(x) - P(x), where:
    - P(x) is the 3D point at time t
    - P'(x) is the transformed point: P' = R * P + t
    
    Args:
        depth1: Depth map at time t (H, W)
        depth2: Depth map at time t+1 (H, W) [optional, currently unused]
        R: Rotation matrix (3, 3)
        t: Translation vector (3, 1) or (3,)
        intrinsics: CameraIntrinsics object
        mask: Optional mask for valid pixels (H, W)
    
    Returns:
        Scene flow (H, W, 3) in camera coordinates [X, Y, Z]
    """
    from ..utils.intrinsics import backproject_depth
    
    # Ensure t is column vector
    if t.ndim == 1:
        t = t.reshape(3, 1)
    
    # Backproject depth to 3D points at time t
    points_t = backproject_depth(depth1, intrinsics, mask=mask)
    
    h, w = depth1.shape
    points_flat = points_t.reshape(-1, 3).T  # (3, N)
    
    # Transform points: P' = R * P + t
    points_t1_flat = R @ points_flat + t  # (3, N)
    points_t1 = points_t1_flat.T.reshape(h, w, 3)
    
    # Scene flow: S = P' - P
    scene_flow = points_t1 - points_t
    
    # Apply mask
    if mask is not None:
        scene_flow = scene_flow * mask[..., None]
    
    return scene_flow


class DepthEstimator:
    """Wrapper for monocular depth estimation."""
    
    def __init__(self, model_name: str = "DPT_Large", device: Optional[str] = None):
        """
        Initialize depth estimator.
        
        Args:
            model_name: Model name ('DPT_Large', 'DPT_Hybrid', 'MiDaS_small', etc.)
            device: Device to run on
        """
        import torch
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        logger.info(f"Loading depth model: {model_name} on {device}")
        
        try:
            # Try to load MiDaS from torch hub
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_name in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            self.available = True
            
        except Exception as e:
            logger.warning(f"Could not load depth model: {e}. Using dummy depth.")
            self.available = False
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from image.
        
        Args:
            image: RGB image (H, W, 3) in [0, 255]
        
        Returns:
            Depth map (H, W) in arbitrary units (inverse depth)
        """
        if not self.available:
            return self._dummy_depth(image)
        
        import torch
        
        # Prepare image
        input_batch = self.transform(image).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth, convert to depth (arbitrary scale)
        depth = 1.0 / (depth + 1e-6)
        
        # Normalize to reasonable range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = depth * 10.0  # Scale to [0, 10] arbitrary units
        
        return depth
    
    def _dummy_depth(self, image: np.ndarray) -> np.ndarray:
        """Generate dummy depth for testing - uses image intensity as proxy."""
        h, w = image.shape[:2]
        
        # Use image brightness as rough depth estimate (darker = farther)
        # This is more realistic than simple radial pattern
        gray = np.mean(image.astype(np.float32), axis=2) / 255.0
        
        # Invert: darker = farther
        depth = 1.0 - gray
        
        # Add some spatial variation
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        radial = np.sqrt((x - cx)**2 + (y - cy)**2) / (max(h, w) / 2)
        
        # Combine: 70% image-based, 30% radial
        depth = 0.7 * depth + 0.3 * radial
        
        # Smooth and scale to reasonable range
        depth = cv2.GaussianBlur(depth, (21, 21), 5.0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = 1.0 + depth * 9.0  # Scale to [1, 10] range
        
        logger.warning("Using dummy depth estimation - results will be approximate. Install PyTorch for MiDaS depth.")
        
        return depth.astype(np.float32)


class SceneFlowEstimator:
    """High-level scene flow estimator using depth + pose."""
    
    def __init__(
        self,
        depth_model: Optional[DepthEstimator] = None,
        intrinsics = None,
        use_weighted_pose: bool = True,
    ):
        """
        Args:
            depth_model: Depth estimation model
            intrinsics: Camera intrinsics
            use_weighted_pose: Whether to use weighted flow for pose estimation
        """
        self.depth_model = depth_model
        self.intrinsics = intrinsics
        self.use_weighted_pose = use_weighted_pose
    
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        flow_result: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Estimate scene flow from two frames and optical flow.
        
        Args:
            image1: First frame (H, W, 3) RGB
            image2: Second frame (H, W, 3) RGB
            flow_result: Dictionary from OpticalFlowEstimator with 'flow' and optionally 'flow_weighted', 'weight'
        
        Returns:
            Dictionary with:
            - 'scene_flow': 3D scene flow (H, W, 3)
            - 'depth1': Depth at time t (H, W)
            - 'depth2': Depth at time t+1 (H, W)
            - 'R': Rotation matrix (3, 3)
            - 't': Translation vector (3, 1)
        """
        # Estimate depth for both frames
        depth1 = self.depth_model(image1)
        depth2 = self.depth_model(image2)
        
        # Select flow for pose estimation
        if self.use_weighted_pose and 'flow_weighted' in flow_result:
            flow_for_pose = flow_result['flow_weighted']
            weight = flow_result.get('weight', None)
        else:
            flow_for_pose = flow_result['flow']
            weight = None
        
        # Estimate pose from flow
        R, t = estimate_pose_from_flow(
            flow_for_pose,
            self.intrinsics,
            weight=weight,
        )
        
        # Compute scene flow
        scene_flow = compute_scene_flow_from_depth_pose(
            depth1,
            depth2,
            R,
            t,
            self.intrinsics,
        )
        
        # Log scene flow statistics for debugging
        sf_mag = np.linalg.norm(scene_flow, axis=-1)
        logger.info(
            f"Scene flow stats: magnitude mean={sf_mag.mean():.4f}, "
            f"max={sf_mag.max():.4f}, std={sf_mag.std():.4f}"
        )
        
        # Check if scene flow is too small (likely identity pose)
        if sf_mag.mean() < 0.01:
            logger.warning(
                "Scene flow magnitude very small - this may indicate:\n"
                "  1. Camera is nearly stationary\n"
                "  2. Pose estimation failed (check optical flow quality)\n"
                "  3. Depth estimation is incorrect"
            )
        
        return {
            'scene_flow': scene_flow,
            'depth1': depth1,
            'depth2': depth2,
            'R': R,
            't': t,
        }

