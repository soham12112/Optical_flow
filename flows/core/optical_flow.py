"""Optical flow estimation with attenuation-aware weighting."""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_optical_flow(
    image1: np.ndarray,
    image2: np.ndarray,
    flow_model,
    weight: Optional[np.ndarray] = None,
    **flow_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optical flow between two frames with optional attenuation weighting.
    
    Args:
        image1: First frame (H, W, 3) RGB in [0, 255]
        image2: Second frame (H, W, 3) RGB in [0, 255]
        flow_model: Flow estimation model (e.g., RAFT)
        weight: Optional weight map (H, W) in [0, 1] for attenuation weighting
        **flow_kwargs: Additional arguments for flow model
    
    Returns:
        Tuple of (flow, weighted_flow)
        - flow: Raw optical flow (H, W, 2)
        - weighted_flow: Attenuation-weighted flow (H, W, 2)
    """
    # Compute raw optical flow
    flow = flow_model(image1, image2, **flow_kwargs)
    
    # Apply attenuation weighting if provided
    if weight is not None:
        weighted_flow = flow * weight[..., None]
    else:
        weighted_flow = flow
    
    logger.debug(
        f"Flow magnitude: raw={np.linalg.norm(flow, axis=-1).mean():.2f}, "
        f"weighted={np.linalg.norm(weighted_flow, axis=-1).mean():.2f}"
    )
    
    return flow, weighted_flow


def compute_forward_backward_consistency(
    flow_forward: np.ndarray,
    flow_backward: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """
    Compute forward-backward consistency mask to detect occlusions.
    
    Args:
        flow_forward: Forward flow from t to t+1 (H, W, 2)
        flow_backward: Backward flow from t+1 to t (H, W, 2)
        threshold: Consistency threshold in pixels
    
    Returns:
        Binary mask (H, W) where 1 indicates consistent (non-occluded) pixels
    """
    h, w = flow_forward.shape[:2]
    
    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w]
    
    # Forward warp: apply forward flow
    x_forward = x + flow_forward[..., 0]
    y_forward = y + flow_forward[..., 1]
    
    # Interpolate backward flow at warped positions
    # Simple nearest-neighbor for now
    x_forward_int = np.clip(np.round(x_forward).astype(int), 0, w - 1)
    y_forward_int = np.clip(np.round(y_forward).astype(int), 0, h - 1)
    
    flow_backward_warped = flow_backward[y_forward_int, x_forward_int]
    
    # Check consistency: flow_forward + flow_backward_warped ≈ 0
    consistency_error = np.linalg.norm(flow_forward + flow_backward_warped, axis=-1)
    
    # Create mask
    mask = (consistency_error < threshold).astype(np.float32)
    
    logger.debug(f"Occlusion mask: {mask.mean()*100:.1f}% consistent pixels")
    
    return mask


def compute_flow_confidence(
    flow: np.ndarray,
    method: str = "magnitude",
) -> np.ndarray:
    """
    Compute per-pixel confidence for optical flow.
    
    Args:
        flow: Optical flow (H, W, 2)
        method: Confidence estimation method
            - 'magnitude': inverse of magnitude (small flow = more confident)
            - 'uniform': uniform confidence
    
    Returns:
        Confidence map (H, W) in [0, 1]
    """
    if method == "uniform":
        return np.ones(flow.shape[:2], dtype=np.float32)
    
    elif method == "magnitude":
        mag = np.linalg.norm(flow, axis=-1)
        # Normalize by percentile
        mag_95 = np.percentile(mag, 95)
        confidence = 1.0 - np.clip(mag / max(mag_95, 1e-6), 0, 1)
        return confidence
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")


class OpticalFlowEstimator:
    """High-level optical flow estimator with preprocessing and weighting."""
    
    def __init__(
        self,
        flow_model,
        use_underwater_weights: bool = True,
        weight_params: Optional[Dict[str, Any]] = None,
        transmission_method: str = "dcp",
    ):
        """
        Args:
            flow_model: Flow model (e.g., RAFT)
            use_underwater_weights: Whether to use attenuation-aware weighting
            weight_params: Parameters for weight_from_transmission
            transmission_method: Method for transmission estimation ('dcp' or 'fast')
        """
        self.flow_model = flow_model
        self.use_underwater_weights = use_underwater_weights
        self.weight_params = weight_params or {"mode": "pow", "gamma": 1.0}
        self.transmission_method = transmission_method
    
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate optical flow with full pipeline.
        
        Args:
            image1: First frame (H, W, 3) RGB in [0, 255]
            image2: Second frame (H, W, 3) RGB in [0, 255]
        
        Returns:
            Dictionary with keys:
            - 'flow': Raw optical flow (H, W, 2)
            - 'flow_weighted': Weighted optical flow (H, W, 2)
            - 'weight': Weight map (H, W) [only if use_underwater_weights=True]
            - 'transmission': Transmission map (H, W) [only if use_underwater_weights=True]
        """
        # Compute attenuation weight if enabled
        weight = None
        transmission = None
        
        if self.use_underwater_weights:
            from .transmission import compute_attenuation_weight
            transmission, weight = compute_attenuation_weight(
                image1,
                method=self.transmission_method,
                weight_params=self.weight_params,
            )
        
        # Compute optical flow
        flow, flow_weighted = compute_optical_flow(
            image1, image2, self.flow_model, weight=weight
        )
        
        # Build result dictionary
        result = {
            'flow': flow,
            'flow_weighted': flow_weighted,
        }
        
        if self.use_underwater_weights:
            result['weight'] = weight
            result['transmission'] = transmission
        
        return result

