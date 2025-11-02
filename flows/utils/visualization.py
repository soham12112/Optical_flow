"""Visualization utilities for optical flow and scene flow."""

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def flow_to_color(
    flow: np.ndarray,
    max_magnitude: Optional[float] = None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert optical flow to color visualization using Sintel color wheel.
    
    Args:
        flow: Optical flow (H, W, 2) with (u, v) components
        max_magnitude: Maximum flow magnitude for saturation (auto if None)
        normalize: Whether to auto-normalize magnitude
    
    Returns:
        RGB color image (H, W, 3) in [0, 255]
    """
    u = flow[..., 0]
    v = flow[..., 1]
    
    # Compute magnitude and angle
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)
    
    # Normalize magnitude
    if max_magnitude is None and normalize:
        max_magnitude = np.percentile(mag, 95)  # Use 95th percentile
        max_magnitude = max(max_magnitude, 1e-6)
    
    if max_magnitude is None:
        max_magnitude = 1.0
    
    # Convert to HSV
    # Hue: angle in [0, 360] -> [0, 180] for OpenCV
    # Saturation: normalized magnitude
    # Value: constant (255)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)  # Hue
    hsv[..., 1] = np.clip(mag / max_magnitude * 255, 0, 255).astype(np.uint8)  # Saturation
    hsv[..., 2] = 255  # Value
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb


def visualize_weight_map(weight: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Visualize weight map with colormap.
    
    Args:
        weight: Weight map (H, W) in [0, 1]
        colormap: OpenCV colormap
    
    Returns:
        RGB visualization (H, W, 3)
    """
    # Scale to [0, 255]
    weight_uint8 = (weight * 255).astype(np.uint8)
    
    # Apply colormap
    colored = cv2.applyColorMap(weight_uint8, colormap)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    return rgb


def scene_flow_to_color(
    scene_flow: np.ndarray,
    mode: str = "magnitude",  # Changed default to magnitude for better visualization
    percentile: float = 95.0,
) -> np.ndarray:
    """
    Visualize 3D scene flow.
    
    Args:
        scene_flow: Scene flow (H, W, 3) with (X, Y, Z) components
        mode: 'xyz' to map X/Y/Z to R/G/B, or 'magnitude' for magnitude heatmap
        percentile: Percentile for clipping outliers
    
    Returns:
        RGB visualization (H, W, 3)
    """
    if mode == "magnitude":
        # Compute magnitude
        mag = np.linalg.norm(scene_flow, axis=-1)
        
        # Check if magnitude is very small
        max_mag = mag.max()
        if max_mag < 1e-6:
            logger.warning("Scene flow magnitude near zero - returning blank visualization")
            return np.zeros((*scene_flow.shape[:2], 3), dtype=np.uint8)
        
        # Normalize using percentile for better contrast
        p_high = np.percentile(mag, percentile)
        if p_high < 1e-6:
            p_high = max_mag
        
        mag_normalized = np.clip(mag / p_high, 0, 1)
        
        # Apply colormap
        mag_uint8 = (mag_normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(mag_uint8, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Add magnitude info overlay (optional)
        mean_mag = mag.mean()
        max_mag = mag.max()
        logger.debug(f"Scene flow visualization: mean_mag={mean_mag:.4f}, max_mag={max_mag:.4f}")
        
    elif mode == "xyz":
        # Map X, Y, Z to R, G, B
        # Use signed normalization (negative = dark, positive = bright)
        vis = np.zeros_like(scene_flow)
        
        for c in range(3):
            channel = scene_flow[..., c]
            
            # Use symmetric range around zero
            abs_max = np.percentile(np.abs(channel), percentile)
            if abs_max < 1e-6:
                vis[..., c] = 0.5  # Neutral gray if no motion
                continue
            
            # Normalize to [-1, 1] then shift to [0, 1]
            normalized = np.clip(channel / abs_max, -1, 1)
            normalized = (normalized + 1.0) / 2.0  # Now in [0, 1]
            
            vis[..., c] = normalized
        
        # Scale to [0, 255]
        rgb = (vis * 255).astype(np.uint8)
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return rgb


def create_flow_legend(size: int = 256) -> np.ndarray:
    """
    Create a color wheel legend for optical flow visualization.
    
    Args:
        size: Size of the legend image (size x size)
    
    Returns:
        RGB legend image
    """
    # Create coordinate grid
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    
    # Create radial flow pattern
    flow = np.stack([x, y], axis=-1)
    
    # Mask to create circular legend
    r = np.sqrt(x**2 + y**2)
    mask = r <= 1.0
    
    # Visualize
    legend = flow_to_color(flow, max_magnitude=1.0, normalize=False)
    
    # Apply mask
    legend[~mask] = 255
    
    return legend


def overlay_flow_vectors(
    image: np.ndarray,
    flow: np.ndarray,
    step: int = 16,
    scale: float = 1.0,
    thickness: int = 1,
) -> np.ndarray:
    """
    Overlay flow vectors on image.
    
    Args:
        image: Background image (H, W, 3)
        flow: Optical flow (H, W, 2)
        step: Spacing between vectors
        scale: Scaling factor for vector length
        thickness: Line thickness
    
    Returns:
        Image with overlaid vectors
    """
    h, w = flow.shape[:2]
    result = image.copy()
    
    # Sample flow vectors
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            u, v = flow[y, x]
            
            # Skip very small flow
            if np.sqrt(u**2 + v**2) < 0.5:
                continue
            
            # Draw arrow
            pt1 = (x, y)
            pt2 = (int(x + u * scale), int(y + v * scale))
            cv2.arrowedLine(result, pt1, pt2, (0, 255, 0), thickness, tipLength=0.3)
    
    return result


def save_video_from_frames(
    frames: list,
    output_path: str,
    fps: float = 24.0,
    codec: str = 'mp4v',
):
    """
    Save a list of frames as video.
    
    Args:
        frames: List of RGB frames (H, W, 3)
        output_path: Output video path
        fps: Frames per second
        codec: Video codec fourcc
    """
    if len(frames) == 0:
        logger.warning("No frames to save")
        return
    
    h, w = frames[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    
    writer.release()
    logger.info(f"Saved video: {output_path} ({len(frames)} frames @ {fps} fps)")

