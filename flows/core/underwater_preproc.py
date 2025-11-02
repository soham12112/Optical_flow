"""Underwater-specific preprocessing: denoising and gamma correction."""

import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """
    Apply gamma correction for low-light enhancement.
    
    Args:
        image: Input RGB image in [0, 255]
        gamma: Gamma value (< 1.0 brightens, typical range 0.7-0.9 for underwater)
    
    Returns:
        Gamma-corrected image in [0, 255]
    """
    # Normalize to [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # Apply gamma
    corrected = np.power(normalized, gamma)
    
    # Back to [0, 255]
    return (corrected * 255.0).astype(np.uint8)


def temporal_denoise(
    image: np.ndarray,
    prev_images: Optional[list] = None,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> np.ndarray:
    """
    Apply temporal denoising using OpenCV's fastNlMeansDenoisingColored.
    
    Args:
        image: Current frame (RGB, uint8)
        prev_images: List of previous frames for temporal denoising (currently unused,
                     OpenCV's function is single-frame but has temporal variant)
        h: Filter strength (higher = more denoising)
        template_window_size: Size of template patch
        search_window_size: Size of search area
    
    Returns:
        Denoised image
    """
    # For single frame denoising
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size
    )
    
    return denoised


def preprocess_underwater_frame(
    image: np.ndarray,
    apply_gamma: bool = True,
    gamma: float = 0.8,
    apply_denoise: bool = True,
    denoise_strength: float = 10.0,
) -> np.ndarray:
    """
    Apply full underwater preprocessing pipeline.
    
    Args:
        image: Input RGB image (uint8, 0-255)
        apply_gamma: Whether to apply gamma correction
        gamma: Gamma value for correction
        apply_denoise: Whether to apply denoising
        denoise_strength: Strength of denoising filter
    
    Returns:
        Preprocessed image
    """
    result = image.copy()
    
    # Apply denoising first (on original sensor data)
    if apply_denoise:
        result = temporal_denoise(result, h=denoise_strength)
        logger.debug("Applied temporal denoising")
    
    # Then gamma correction
    if apply_gamma:
        result = apply_gamma_correction(result, gamma=gamma)
        logger.debug(f"Applied gamma correction (γ={gamma})")
    
    return result


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
    Alternative to gamma correction for local contrast enhancement.
    
    Args:
        image: Input RGB image (uint8)
        clip_limit: CLAHE clip limit
        tile_size: Grid size for CLAHE
    
    Returns:
        Enhanced image
    """
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return rgb_enhanced

