"""
RAFT optical flow model wrapper.
Requires the official RAFT implementation.
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class RAFTWrapper:
    """Wrapper for RAFT optical flow model."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        small: bool = False,
    ):
        """
        Initialize RAFT model.
        
        Args:
            model_path: Path to RAFT weights (.pth file)
            device: 'cuda' or 'cpu' (auto-detect if None)
            small: Use RAFT-small variant
        """
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        logger.info(f"Initializing RAFT on {self.device}")
        
        # Try to import RAFT
        try:
            # Attempt to import from installed RAFT package or local directory
            from .raft_core.raft import RAFT
            from .raft_core.utils.utils import InputPadder
            self.InputPadder = InputPadder
        except ImportError:
            logger.error(
                "RAFT not found. Please clone and setup RAFT:\n"
                "  git clone https://github.com/princeton-vl/RAFT.git flows/models/raft_core\n"
                "  Or install: pip install git+https://github.com/princeton-vl/RAFT.git"
            )
            raise
        
        # Initialize model
        args = type('Args', (), {
            'small': small,
            'mixed_precision': False,
            'alternate_corr': False,
        })()
        
        self.model = RAFT(args)
        
        # Load weights if provided
        if model_path is not None:
            self._load_weights(model_path)
        else:
            logger.warning("No weights provided, using random initialization")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_weights(self, path: str):
        """Load model weights."""
        logger.info(f"Loading RAFT weights from {path}")
        state_dict = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        self.model.load_state_dict(state_dict)
    
    @torch.no_grad()
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        iters: int = 20,
    ) -> np.ndarray:
        """
        Compute optical flow between two images.
        
        Args:
            image1: First image (H, W, 3) RGB in [0, 255]
            image2: Second image (H, W, 3) RGB in [0, 255]
            iters: Number of refinement iterations
        
        Returns:
            Optical flow (H, W, 2) with (u, v) components
        """
        # Convert to torch tensors
        img1_torch = self._prepare_image(image1)
        img2_torch = self._prepare_image(image2)
        
        # Pad images
        padder = self.InputPadder(img1_torch.shape)
        img1_torch, img2_torch = padder.pad(img1_torch, img2_torch)
        
        # Compute flow
        _, flow_up = self.model(img1_torch, img2_torch, iters=iters, test_mode=True)
        
        # Unpad
        flow_up = padder.unpad(flow_up)
        
        # Convert to numpy
        flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()
        
        return flow_np
    
    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to torch tensor for RAFT."""
        # Convert to float and normalize to [0, 1]
        img = torch.from_numpy(image).permute(2, 0, 1).float()
        img = img[None].to(self.device)  # Add batch dimension
        return img


class DummyRAFT:
    """Dummy RAFT for testing without actual model."""
    
    def __init__(self, **kwargs):
        logger.warning("Using DummyRAFT - producing random flow for testing")
        self.device = torch.device('cpu')
    
    def __call__(self, image1: np.ndarray, image2: np.ndarray, **kwargs) -> np.ndarray:
        """Generate random flow."""
        h, w = image1.shape[:2]
        
        # Simple correlation-based flow for demonstration
        # In practice, this should be replaced with actual RAFT
        gray1 = np.mean(image1, axis=2).astype(np.float32)
        gray2 = np.mean(image2, axis=2).astype(np.float32)
        
        # Compute simple gradient-based flow
        diff = gray2 - gray1
        gy, gx = np.gradient(gray1)
        
        # Rough estimate (not accurate!)
        u = np.where(np.abs(gx) > 1e-6, -diff / (gx + 1e-6), 0)
        v = np.where(np.abs(gy) > 1e-6, -diff / (gy + 1e-6), 0)
        
        # Clip to reasonable range
        u = np.clip(u, -20, 20)
        v = np.clip(v, -20, 20)
        
        flow = np.stack([u, v], axis=-1)
        
        return flow.astype(np.float32)


def create_raft_model(
    backend: str = "raft",
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> RAFTWrapper:
    """
    Factory function to create RAFT model.
    
    Args:
        backend: Model backend ('raft', 'raft-small', 'dummy')
        model_path: Path to model weights
        device: Device to run on
    
    Returns:
        RAFT model wrapper
    """
    if backend == "dummy":
        return DummyRAFT()
    elif backend == "raft":
        return RAFTWrapper(model_path=model_path, device=device, small=False)
    elif backend == "raft-small":
        return RAFTWrapper(model_path=model_path, device=device, small=True)
    else:
        raise ValueError(f"Unknown backend: {backend}")

