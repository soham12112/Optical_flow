"""Frame extraction from video files."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    resize_long_edge: Optional[int] = None,
    fps_out: Optional[float] = None,
    stride: int = 1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to input video file
        resize_long_edge: If set, resize frames so longest edge equals this value
        fps_out: If set, resample to this fps (otherwise use original fps)
        stride: Frame stride (1=every frame, 2=every other frame, etc.)
    
    Yields:
        Tuple of (frame_index, frame_rgb)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height}, {fps_in:.2f} fps, {total_frames} frames")
    
    # Calculate frame skip for fps_out
    frame_skip = 1
    if fps_out is not None and fps_in > fps_out:
        frame_skip = int(round(fps_in / fps_out))
        logger.info(f"Resampling from {fps_in:.2f} to ~{fps_in/frame_skip:.2f} fps")
    
    # Calculate resize dimensions
    resize_dims = None
    if resize_long_edge is not None:
        if width > height:
            new_w = resize_long_edge
            new_h = int(height * resize_long_edge / width)
        else:
            new_h = resize_long_edge
            new_w = int(width * resize_long_edge / height)
        resize_dims = (new_w, new_h)
        logger.info(f"Resizing from {width}x{height} to {new_w}x{new_h}")
    
    frame_idx = 0
    extracted_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on fps_out and stride
            if frame_idx % (frame_skip * stride) != 0:
                frame_idx += 1
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize if requested
            if resize_dims is not None:
                frame_rgb = cv2.resize(frame_rgb, resize_dims, interpolation=cv2.INTER_LINEAR)
            
            yield extracted_idx, frame_rgb
            extracted_idx += 1
            frame_idx += 1
            
    finally:
        cap.release()
    
    logger.info(f"Extracted {extracted_idx} frames")


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Returns:
        Dictionary with fps, width, height, frame_count
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    
    cap.release()
    return info

