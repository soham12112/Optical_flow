"""
Transmission map estimation and attenuation-aware weighting.
Implements normalized medium transmission maps as described in wflow-TartanVO.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute dark channel prior.
    
    Args:
        image: RGB image in [0, 255] or [0, 1]
        patch_size: Size of local patch
    
    Returns:
        Dark channel map
    """
    # Get minimum across color channels
    min_channel = np.min(image, axis=2)
    
    # Apply minimum filter (erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_ch = cv2.erode(min_channel, kernel)
    
    return dark_ch


def estimate_background_light(image: np.ndarray, dark_ch: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Estimate background light (veiling light in underwater imaging).
    Uses brightest pixels in dark channel.
    
    Args:
        image: RGB image in [0, 255]
        dark_ch: Pre-computed dark channel (optional)
    
    Returns:
        Background light vector [R, G, B]
    """
    if dark_ch is None:
        dark_ch = dark_channel(image)
    
    # Get top 0.1% brightest pixels in dark channel
    num_pixels = dark_ch.size
    num_brightest = max(int(num_pixels * 0.001), 1)
    
    # Flatten and find brightest indices
    dark_flat = dark_ch.flatten()
    indices = np.argpartition(dark_flat, -num_brightest)[-num_brightest:]
    
    # Get corresponding pixels in original image
    h, w = dark_ch.shape
    image_flat = image.reshape(-1, 3)
    brightest_pixels = image_flat[indices]
    
    # Background light is the pixel with maximum intensity
    intensities = np.sum(brightest_pixels, axis=1)
    max_idx = np.argmax(intensities)
    B = brightest_pixels[max_idx]
    
    return B.astype(np.float32)


def estimate_transmission_underwater(
    image: np.ndarray,
    B: Optional[np.ndarray] = None,
    omega: float = 0.95,
    patch_size: int = 15,
    guided_filter: bool = True,
) -> np.ndarray:
    """
    Estimate transmission map using underwater-adapted dark channel prior.
    
    This implements a simplified version of underwater transmission estimation.
    The transmission t(x) = exp(-β * d(x)) represents medium clarity.
    
    Args:
        image: RGB image in [0, 255]
        B: Background light (if None, will be estimated)
        omega: Retention parameter (typically 0.85-0.95)
        patch_size: Patch size for dark channel
        guided_filter: Whether to refine with guided filter
    
    Returns:
        Transmission map in [0, 1]
    """
    # Normalize image
    img_norm = image.astype(np.float32) / 255.0
    
    # Estimate background light if not provided
    if B is None:
        B = estimate_background_light(image)
        B = B / 255.0
    else:
        B = B / 255.0 if B.max() > 1.0 else B
    
    # Avoid division by zero
    B = np.maximum(B, 1e-6)
    
    # Compute transmission using dark channel
    # I(x) = J(x) * t(x) + B * (1 - t(x))
    # Rearranging: t(x) ≈ 1 - omega * dark_channel(I(x) / B)
    
    I_normalized = img_norm / B
    I_normalized = np.clip(I_normalized, 0, 1)
    
    dark_ch = dark_channel(I_normalized, patch_size=patch_size)
    transmission = 1.0 - omega * dark_ch
    
    # Clip to valid range
    transmission = np.clip(transmission, 0.1, 1.0)
    
    # Refine with guided filter to preserve edges
    if guided_filter:
        gray = cv2.cvtColor((image).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        transmission = cv2.ximgproc.guidedFilter(
            gray.astype(np.float32),
            transmission.astype(np.float32),
            radius=40,
            eps=1e-3
        )
        transmission = np.clip(transmission, 0.1, 1.0)
    
    return transmission


def estimate_transmission_fast(image: np.ndarray) -> np.ndarray:
    """
    Fast approximation of transmission map using simple heuristics.
    Useful for real-time processing.
    
    Args:
        image: RGB image in [0, 255]
    
    Returns:
        Transmission map in [0, 1]
    """
    # Convert to LAB
    lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.float32) / 255.0
    
    # Use luminance as proxy for transmission (brighter = more backscatter = lower transmission)
    # Invert: darker regions (less backscatter) have higher transmission
    transmission = 1.0 - l_channel
    
    # Apply smoothing
    transmission = cv2.GaussianBlur(transmission, (31, 31), 10)
    
    # Normalize to reasonable range
    transmission = np.clip(transmission, 0.2, 1.0)
    
    # Normalize to [0, 1]
    t_min, t_max = transmission.min(), transmission.max()
    if t_max > t_min:
        transmission = (transmission - t_min) / (t_max - t_min)
    
    return transmission


def weight_from_transmission(
    transmission: np.ndarray,
    mode: str = "pow",
    gamma: float = 1.0,
    a: float = 1.0,
    b: float = 0.0,
) -> np.ndarray:
    """
    Convert transmission map to weight map for flow.
    
    From wflow-TartanVO: w(x) emphasizes clearer regions (high t(x))
    and suppresses hazy regions (low t(x)).
    
    Args:
        transmission: Transmission map in [0, 1]
        mode: 'pow' for power function, 'linear' for linear scaling
        gamma: Power exponent (γ > 1 emphasizes high transmission more)
        a: Linear scaling factor
        b: Linear bias
    
    Returns:
        Weight map in [0, 1]
    """
    if mode == "pow":
        weight = np.power(transmission, gamma)
    elif mode == "linear":
        weight = a * transmission + b
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    
    weight = np.clip(weight, 0.0, 1.0)
    
    return weight


def compute_attenuation_weight(
    image: np.ndarray,
    method: str = "dcp",
    weight_params: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete pipeline: estimate transmission and compute attenuation-aware weights.
    
    Args:
        image: RGB image in [0, 255]
        method: 'dcp' for dark channel prior, 'fast' for fast approximation
        weight_params: Parameters for weight_from_transmission
    
    Returns:
        Tuple of (transmission_map, weight_map), both in [0, 1]
    """
    if weight_params is None:
        weight_params = {"mode": "pow", "gamma": 1.0}
    
    # Estimate transmission
    if method == "dcp":
        transmission = estimate_transmission_underwater(image, guided_filter=True)
    elif method == "fast":
        transmission = estimate_transmission_fast(image)
    else:
        raise ValueError(f"Unknown transmission estimation method: {method}")
    
    # Compute weights
    weight = weight_from_transmission(transmission, **weight_params)
    
    logger.debug(f"Transmission range: [{transmission.min():.3f}, {transmission.max():.3f}]")
    logger.debug(f"Weight range: [{weight.min():.3f}, {weight.max():.3f}]")
    
    return transmission, weight

