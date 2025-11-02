"""Command-line interface for underwater flow estimation."""

import argparse
import logging
import sys
from pathlib import Path

from .core.processor import process_video
from .utils.intrinsics import load_intrinsics


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Underwater Optical Flow and Scene Flow with Attenuation-Aware Weighting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (.mp4)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    
    # Models
    parser.add_argument(
        "--backend",
        type=str,
        default="dummy",
        choices=["raft", "raft-small", "dummy"],
        help="Optical flow backend (use 'dummy' for testing without RAFT)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to RAFT model weights (.pth)",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="Depth estimation model",
    )
    
    # Scene flow
    parser.add_argument(
        "--scene_flow_track",
        type=str,
        default="depthpose",
        choices=["depthpose", "none"],
        help="Scene flow estimation method",
    )
    
    # Camera intrinsics
    parser.add_argument(
        "--intrinsics_json",
        type=str,
        default=None,
        help="Path to camera intrinsics JSON file",
    )
    
    # Underwater weighting
    parser.add_argument(
        "--use_underwater_weights",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable attenuation-aware weighting (1=yes, 0=no)",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default="pow",
        choices=["pow", "linear"],
        help="Weight function mode",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma parameter for weight function (mode=pow)",
    )
    parser.add_argument(
        "--transmission_method",
        type=str,
        default="fast",
        choices=["dcp", "fast"],
        help="Transmission map estimation method (dcp=dark channel prior, fast=quick approximation)",
    )
    
    # Preprocessing
    parser.add_argument(
        "--preprocess",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply underwater preprocessing (denoising + gamma correction)",
    )
    parser.add_argument(
        "--preprocess_gamma",
        type=float,
        default=0.8,
        help="Gamma value for preprocessing (< 1.0 brightens)",
    )
    parser.add_argument(
        "--denoise_strength",
        type=float,
        default=10.0,
        help="Denoising filter strength",
    )
    
    # Video processing
    parser.add_argument(
        "--resize_long_edge",
        type=int,
        default=960,
        help="Resize frames so longest edge equals this value (None=no resize)",
    )
    parser.add_argument(
        "--fps_out",
        type=float,
        default=None,
        help="Output FPS (None=use original)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride (1=every frame, 2=every other, etc.)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", None],
        help="Device to run on (None=auto-detect)",
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    # Load intrinsics if provided
    intrinsics = None
    if args.intrinsics_json is not None:
        try:
            intrinsics = load_intrinsics(args.intrinsics_json)
            logger.info(f"Loaded intrinsics from {args.intrinsics_json}")
        except Exception as e:
            logger.error(f"Failed to load intrinsics: {e}")
            sys.exit(1)
    
    # Prepare weight parameters
    weight_params = {
        "mode": args.weight_mode,
        "gamma": args.gamma,
    }
    
    # Run processing
    try:
        info = process_video(
            video_path=args.video,
            out_dir=args.out_dir,
            backend=args.backend,
            model_path=args.weights,
            scene_flow_track=args.scene_flow_track,
            intrinsics=intrinsics,
            use_underwater_weights=bool(args.use_underwater_weights),
            weight_params=weight_params,
            transmission_method=args.transmission_method,
            resize_long_edge=args.resize_long_edge if args.resize_long_edge > 0 else None,
            fps_out=args.fps_out,
            stride=args.stride,
            preprocess_underwater=bool(args.preprocess),
            gamma=args.preprocess_gamma,
            denoise_strength=args.denoise_strength,
            depth_model_name=args.depth_model,
            device=args.device,
        )
        
        logger.info("=" * 60)
        logger.info("Processing complete!")
        logger.info(f"Processed {info['num_frames_processed']} frames")
        logger.info(f"Generated {info['num_flow_pairs']} flow pairs")
        logger.info(f"Outputs saved to: {args.out_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

