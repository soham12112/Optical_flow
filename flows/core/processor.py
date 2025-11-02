"""Main processing pipeline for video flow estimation."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import json

from .frame_extraction import extract_frames, get_video_info
from .underwater_preproc import preprocess_underwater_frame
from .optical_flow import OpticalFlowEstimator
from .scene_flow import SceneFlowEstimator, DepthEstimator
from ..utils.intrinsics import CameraIntrinsics, estimate_intrinsics
from ..utils.visualization import (
    flow_to_color, scene_flow_to_color, visualize_weight_map,
    save_video_from_frames, create_flow_legend
)
from ..models.raft_wrapper import create_raft_model

logger = logging.getLogger(__name__)


def process_video(
    video_path: str,
    out_dir: str = "./outputs",
    backend: str = "raft",
    model_path: Optional[str] = None,
    scene_flow_track: str = "depthpose",
    intrinsics: Optional[CameraIntrinsics] = None,
    use_underwater_weights: bool = True,
    weight_params: Optional[Dict[str, Any]] = None,
    transmission_method: str = "fast",
    resize_long_edge: Optional[int] = 960,
    fps_out: Optional[float] = None,
    stride: int = 1,
    preprocess_underwater: bool = True,
    gamma: float = 0.8,
    denoise_strength: float = 10.0,
    depth_model_name: str = "DPT_Large",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process video to extract optical flow and scene flow with attenuation-aware weighting.
    
    Args:
        video_path: Path to input video
        out_dir: Output directory
        backend: Flow backend ('raft', 'raft-small', 'dummy')
        model_path: Path to RAFT weights (optional)
        scene_flow_track: 'depthpose' or 'mono' (mono not yet implemented)
        intrinsics: Camera intrinsics (will estimate if None)
        use_underwater_weights: Enable attenuation-aware weighting
        weight_params: Parameters for weight function
        transmission_method: 'dcp' or 'fast'
        resize_long_edge: Resize frames to this long edge (None to keep original)
        fps_out: Output fps (None to use original)
        stride: Frame stride (1=every frame)
        preprocess_underwater: Apply underwater preprocessing
        gamma: Gamma correction value
        denoise_strength: Denoising strength
        depth_model_name: Depth model name for scene flow
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with processing statistics and output paths
    """
    # Setup output directories
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    flow_raw_dir = out_path / "optical_flow" / "flow_raw"
    flow_vis_dir = out_path / "optical_flow"
    scene_flow_raw_dir = out_path / "scene_flow" / "scene_flow_raw"
    scene_flow_vis_dir = out_path / "scene_flow"
    diagnostics_dir = out_path / "diagnostics"
    
    for d in [flow_raw_dir, scene_flow_raw_dir, diagnostics_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {out_dir}")
    
    # Get video info
    video_info = get_video_info(video_path)
    logger.info(f"Video info: {video_info}")
    
    # Setup intrinsics
    if intrinsics is None:
        width = video_info['width']
        height = video_info['height']
        if resize_long_edge is not None:
            if width > height:
                width, height = resize_long_edge, int(height * resize_long_edge / width)
            else:
                width, height = int(width * resize_long_edge / height), resize_long_edge
        intrinsics = estimate_intrinsics(width, height)
    
    logger.info(f"Camera intrinsics: {intrinsics}")
    
    # Save intrinsics
    with open(out_path / "intrinsics.json", 'w') as f:
        json.dump(intrinsics.to_dict(), f, indent=2)
    
    # Initialize models
    logger.info("Initializing models...")
    
    # Optical flow model
    flow_model = create_raft_model(backend=backend, model_path=model_path, device=device)
    
    # Optical flow estimator
    flow_estimator = OpticalFlowEstimator(
        flow_model=flow_model,
        use_underwater_weights=use_underwater_weights,
        weight_params=weight_params or {"mode": "pow", "gamma": 1.0},
        transmission_method=transmission_method,
    )
    
    # Scene flow estimator (if enabled)
    scene_flow_estimator = None
    if scene_flow_track == "depthpose":
        depth_model = DepthEstimator(model_name=depth_model_name, device=device)
        scene_flow_estimator = SceneFlowEstimator(
            depth_model=depth_model,
            intrinsics=intrinsics,
            use_weighted_pose=use_underwater_weights,
        )
    elif scene_flow_track == "none":
        logger.info("Scene flow estimation disabled")
    else:
        raise ValueError(f"Unknown scene_flow_track: {scene_flow_track}")
    
    # Process frames
    logger.info("Extracting and processing frames...")
    
    frames = []
    flow_vis_frames = []
    scene_flow_vis_frames = []
    weight_vis_frames = []
    transmission_vis_frames = []
    
    prev_frame = None
    prev_idx = None
    
    frame_iterator = extract_frames(
        video_path,
        resize_long_edge=resize_long_edge,
        fps_out=fps_out,
        stride=stride,
    )
    
    for idx, frame in tqdm(frame_iterator, desc="Processing frames"):
        # Preprocess underwater frame
        if preprocess_underwater:
            frame_processed = preprocess_underwater_frame(
                frame,
                apply_gamma=True,
                gamma=gamma,
                apply_denoise=True,
                denoise_strength=denoise_strength,
            )
        else:
            frame_processed = frame
        
        frames.append(frame_processed)
        
        # Process pairs
        if prev_frame is not None:
            logger.debug(f"Processing frame pair {prev_idx}-{idx}")
            
            # Optical flow
            flow_result = flow_estimator(prev_frame, frame_processed)
            
            # Save raw flow
            flow_file = flow_raw_dir / f"flow_{prev_idx:06d}_{idx:06d}.npy"
            np.save(flow_file, flow_result['flow'])
            
            # Visualize flow
            flow_vis = flow_to_color(flow_result['flow_weighted'])
            flow_vis_frames.append(flow_vis)
            
            # Save diagnostics
            if use_underwater_weights and 'weight' in flow_result:
                weight_vis = visualize_weight_map(flow_result['weight'])
                weight_vis_frames.append(weight_vis)
                
                transmission_vis = visualize_weight_map(flow_result['transmission'])
                transmission_vis_frames.append(transmission_vis)
                
                # Save raw weight and transmission
                np.save(diagnostics_dir / f"weight_{prev_idx:06d}.npy", flow_result['weight'])
                np.save(diagnostics_dir / f"transmission_{prev_idx:06d}.npy", flow_result['transmission'])
            
            # Scene flow
            if scene_flow_estimator is not None:
                try:
                    scene_flow_result = scene_flow_estimator(prev_frame, frame_processed, flow_result)
                    
                    # Save raw scene flow
                    scene_flow_file = scene_flow_raw_dir / f"scene_flow_{prev_idx:06d}_{idx:06d}.npy"
                    np.save(scene_flow_file, scene_flow_result['scene_flow'])
                    
                    # Visualize scene flow (magnitude mode for better visualization)
                    scene_flow_vis = scene_flow_to_color(scene_flow_result['scene_flow'], mode="magnitude")
                    scene_flow_vis_frames.append(scene_flow_vis)
                    
                    # Save depth maps
                    np.save(diagnostics_dir / f"depth_{prev_idx:06d}.npy", scene_flow_result['depth1'])
                    
                    # Save pose info
                    pose_info = {
                        'R': scene_flow_result['R'].tolist(),
                        't': scene_flow_result['t'].tolist(),
                    }
                    with open(diagnostics_dir / f"pose_{prev_idx:06d}_{idx:06d}.json", 'w') as f:
                        json.dump(pose_info, f, indent=2)
                
                except Exception as e:
                    logger.error(f"Scene flow estimation failed for pair {prev_idx}-{idx}: {e}")
                    # Create blank frame to maintain video sync
                    if len(scene_flow_vis_frames) > 0:
                        scene_flow_vis_frames.append(np.zeros_like(scene_flow_vis_frames[-1]))
                    else:
                        scene_flow_vis_frames.append(np.zeros((frame_processed.shape[0], frame_processed.shape[1], 3), dtype=np.uint8))
        
        prev_frame = frame_processed
        prev_idx = idx
    
    logger.info(f"Processed {len(frames)} frames, {len(flow_vis_frames)} flow pairs")
    
    # Save videos
    output_fps = fps_out if fps_out is not None else video_info['fps']
    
    if len(flow_vis_frames) > 0:
        logger.info("Saving optical flow video...")
        save_video_from_frames(
            flow_vis_frames,
            str(flow_vis_dir / "flow_vis.mp4"),
            fps=output_fps,
        )
        
        # Save flow legend
        legend = create_flow_legend(size=256)
        import cv2
        cv2.imwrite(str(diagnostics_dir / "flow_legend.png"), cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))
    
    if len(scene_flow_vis_frames) > 0:
        logger.info("Saving scene flow video...")
        save_video_from_frames(
            scene_flow_vis_frames,
            str(scene_flow_vis_dir / "scene_flow_vis.mp4"),
            fps=output_fps,
        )
    
    if len(weight_vis_frames) > 0:
        logger.info("Saving diagnostic videos...")
        save_video_from_frames(
            weight_vis_frames,
            str(diagnostics_dir / "weights.mp4"),
            fps=output_fps,
        )
        save_video_from_frames(
            transmission_vis_frames,
            str(diagnostics_dir / "transmission.mp4"),
            fps=output_fps,
        )
    
    # Save processing info
    info = {
        "video_path": video_path,
        "video_info": video_info,
        "num_frames_processed": len(frames),
        "num_flow_pairs": len(flow_vis_frames),
        "intrinsics": intrinsics.to_dict(),
        "config": {
            "backend": backend,
            "use_underwater_weights": use_underwater_weights,
            "weight_params": weight_params,
            "transmission_method": transmission_method,
            "scene_flow_track": scene_flow_track,
            "preprocess_underwater": preprocess_underwater,
            "gamma": gamma,
            "denoise_strength": denoise_strength,
        }
    }
    
    with open(out_path / "processing_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info("Processing complete!")
    logger.info(f"Outputs saved to: {out_dir}")
    
    return info

