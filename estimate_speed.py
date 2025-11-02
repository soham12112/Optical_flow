#!/usr/bin/env python3
"""
Example: Estimate animal speed from optical flow using analytical motion field equations.

This script:
1. Loads optical flow from a processed video
2. Estimates relative speed using motion field equations (no depth required)
3. Calibrates scale factor from CSV ground-truth speeds
4. Outputs metric speed estimates (m/s) for all frames
"""

import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from flows.core.speed_estimation import (
    AnalyticalSpeedEstimator,
    load_speed_csv,
    smooth_speeds,
)
from flows.utils.intrinsics import CameraIntrinsics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_flow_results(output_dir: str) -> dict:
    """Load optical flow results from output directory."""
    output_path = Path(output_dir)
    
    # Load processing info to get frame info
    import json
    
    # Try both possible filenames
    metadata_file = output_path / "processing_info.json"
    if not metadata_file.exists():
        metadata_file = output_path / "metadata.json"  # Fallback for older versions
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Processing info not found: {output_path}/processing_info.json or metadata.json")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Extract frame info (handles both old and new format)
    n_frames = metadata.get('num_frames_processed', metadata.get('num_frames', 0))
    
    # Get FPS from video_info if available, otherwise top-level
    if 'video_info' in metadata:
        fps = metadata['video_info'].get('fps', 30.0)
    else:
        fps = metadata.get('fps', 30.0)
    
    logger.info(f"Loading flow results: {n_frames} frames at {fps} fps")
    
    # Load flow and weights
    # The processor saves to:
    #   - optical_flow/flow_raw/flow_XXXXXX_YYYYYY.npy
    #   - diagnostics/weight_XXXXXX.npy
    flows = []
    weights = []
    frame_times = []
    
    for i in range(n_frames - 1):  # Flow is computed between consecutive frames
        # Try new format first (optical_flow/flow_raw/), then old format (flow/)
        flow_file = output_path / "optical_flow" / "flow_raw" / f"flow_{i:06d}_{i+1:06d}.npy"
        if not flow_file.exists():
            flow_file = output_path / "flow" / f"flow_{i:06d}.npy"
        
        # Try diagnostics/ for weights, then weights/
        weight_file = output_path / "diagnostics" / f"weight_{i:06d}.npy"
        if not weight_file.exists():
            weight_file = output_path / "weights" / f"weight_{i:06d}.npy"
        
        if not flow_file.exists():
            logger.warning(f"Flow file not found: {flow_file}")
            continue
        
        flow = np.load(flow_file)
        flows.append(flow)
        
        if weight_file.exists():
            weight = np.load(weight_file)
        else:
            logger.warning(f"Weight file not found: {weight_file}, using uniform weights")
            weight = np.ones(flow.shape[:2], dtype=np.float32)
        weights.append(weight)
        
        # Frame time is at the start of the interval
        frame_times.append(i / fps)
    
    logger.info(f"Loaded {len(flows)} flow fields")
    
    return {
        'flows': flows,
        'weights': weights,
        'frame_times': np.array(frame_times),
        'fps': fps,
        'metadata': metadata,
    }


def estimate_speeds_from_flow(
    flow_results: dict,
    intrinsics: CameraIntrinsics,
    csv_path: Optional[str] = None,
    csv_time_col: str = "time",
    csv_speed_col: str = "speed",
    smooth_output: bool = True,
    smooth_window: int = 5,
    time_offset: Optional[float] = None,
) -> dict:
    """
    Estimate speeds from optical flow results.
    
    Args:
        flow_results: Dictionary from load_flow_results
        intrinsics: Camera intrinsics
        csv_path: Optional path to CSV with ground-truth speeds for calibration
        csv_time_col: Name of time column in CSV
        csv_speed_col: Name of speed column in CSV
        smooth_output: Whether to smooth speed estimates
        smooth_window: Window size for smoothing
    
    Returns:
        Dictionary with speed estimates and diagnostics
    """
    # Create speed estimator
    estimator = AnalyticalSpeedEstimator(
        intrinsics=intrinsics,
        use_robust_estimation=True,
        min_pixels=100,
    )
    
    flows = flow_results['flows']
    weights = flow_results['weights']
    frame_times = flow_results['frame_times']
    fps = flow_results['fps']
    dt = 1.0 / fps
    
    # Estimate speed for each frame
    logger.info("Estimating speeds from optical flow...")
    
    speed_estimates = []
    relative_speeds = []
    confidences = []
    omegas = []
    
    for i, (flow, weight) in enumerate(zip(flows, weights)):
        estimate = estimator.estimate_speed(flow, weight, dt=dt)
        
        speed_estimates.append(estimate)
        relative_speeds.append(estimate.rel_speed)
        confidences.append(estimate.confidence)
        omegas.append(estimate.omega)
        
        if i % 50 == 0:
            logger.info(
                f"Frame {i}/{len(flows)}: rel_speed={estimate.rel_speed:.4f}, "
                f"confidence={estimate.confidence:.3f}, pixels={estimate.n_pixels_used}"
            )
    
    relative_speeds = np.array(relative_speeds)
    confidences = np.array(confidences)
    omegas = np.array(omegas)
    
    # Calibrate scale factor if CSV provided
    if csv_path is not None:
        logger.info(f"Calibrating scale factor from CSV: {csv_path}")
        csv_times, csv_speeds = load_speed_csv(csv_path, csv_time_col, csv_speed_col)
        
        scale_factor = estimator.calibrate_scale(
            frame_times=frame_times,
            relative_speeds=relative_speeds,
            csv_times=csv_times,
            csv_speeds=csv_speeds,
            use_robust=True,
            time_offset=time_offset,
        )
        
        # Convert to metric speeds
        metric_speeds = scale_factor * relative_speeds
        
        # Store both original and aligned CSV times for reference
        csv_times_aligned = csv_times - (csv_times[0] - frame_times[0]) if time_offset is None else csv_times - time_offset
        
        result = {
            'frame_times': frame_times,
            'relative_speeds': relative_speeds,
            'metric_speeds': metric_speeds,
            'scale_factor': scale_factor,
            'calibrated': True,
            'csv_times': csv_times_aligned,  # Use aligned times for plotting
            'csv_times_original': csv_times,  # Keep original for reference
            'csv_speeds': csv_speeds,
        }
    else:
        logger.warning("No CSV provided - speeds are in relative units")
        metric_speeds = relative_speeds
        
        result = {
            'frame_times': frame_times,
            'relative_speeds': relative_speeds,
            'metric_speeds': metric_speeds,
            'scale_factor': 1.0,
            'calibrated': False,
        }
    
    # Smooth speeds if requested
    if smooth_output and len(metric_speeds) > smooth_window:
        logger.info(f"Smoothing speeds with window={smooth_window}")
        metric_speeds_smooth = smooth_speeds(metric_speeds, smooth_window, method='savgol')
        result['metric_speeds_smooth'] = metric_speeds_smooth
        result['metric_speeds_raw'] = metric_speeds
        result['metric_speeds'] = metric_speeds_smooth
    
    # Add diagnostics
    result['confidences'] = confidences
    result['omegas'] = omegas
    result['speed_estimates'] = speed_estimates
    
    return result


def plot_speed_comparison(
    result: dict,
    output_path: Optional[str] = None,
):
    """Plot speed estimates vs CSV ground truth."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    frame_times = result['frame_times']
    metric_speeds = result['metric_speeds']
    confidences = result['confidences']
    
    # Plot 1: Speed comparison
    ax = axes[0]
    ax.plot(frame_times, metric_speeds, 'b-', linewidth=2, label='Estimated speed')
    
    if 'metric_speeds_raw' in result:
        ax.plot(frame_times, result['metric_speeds_raw'], 'b--', alpha=0.3, label='Raw (before smoothing)')
    
    if result['calibrated']:
        csv_times = result['csv_times']  # These are already aligned
        csv_speeds = result['csv_speeds']
        ax.plot(csv_times, csv_speeds, 'r-', linewidth=2, label='CSV ground truth')
        
        # Compute error
        from scipy.interpolate import interp1d
        interp_csv = interp1d(csv_times, csv_speeds, kind='linear', bounds_error=False, fill_value='extrapolate')
        csv_at_frames = interp_csv(frame_times)
        rmse = np.sqrt(np.mean((metric_speeds - csv_at_frames)**2))
        mae = np.mean(np.abs(metric_speeds - csv_at_frames))
        
        # Show time alignment info if available
        time_info = ""
        if 'csv_times_original' in result:
            offset = result['csv_times_original'][0] - csv_times[0]
            if abs(offset) > 0.1:
                time_info = f" (CSV offset: {offset:.1f}s)"
        
        ax.set_title(f'Speed Estimation (RMSE={rmse:.3f} m/s, MAE={mae:.3f} m/s){time_info}')
        ax.set_ylabel('Speed (m/s)')
    else:
        ax.set_title('Speed Estimation (Relative Units - No Calibration)')
        ax.set_ylabel('Relative speed (1/s)')
    
    ax.set_xlabel('Time (seconds from video start)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Confidence scores
    ax = axes[1]
    ax.plot(frame_times, confidences, 'g-', linewidth=2)
    ax.set_xlabel('Time (seconds from video start)')
    ax.set_ylabel('Confidence')
    ax.set_title('Estimation Confidence')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Angular velocity (rotation)
    ax = axes[2]
    omegas = result['omegas']
    ax.plot(frame_times, omegas[:, 0], 'r-', label='ωx (pitch rate)', alpha=0.7)
    ax.plot(frame_times, omegas[:, 1], 'g-', label='ωy (yaw rate)', alpha=0.7)
    ax.plot(frame_times, omegas[:, 2], 'b-', label='ωz (roll rate)', alpha=0.7)
    ax.set_xlabel('Time (seconds from video start)')
    ax.set_ylabel('Angular velocity (rad/s)')
    ax.set_title('Estimated Camera Rotation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()


def save_speed_results(
    result: dict,
    output_path: str,
):
    """Save speed estimation results to NPY and CSV."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as NPY (full data)
    np.save(output_path / "speed_results.npy", result)
    logger.info(f"Saved results to {output_path / 'speed_results.npy'}")
    
    # Save as CSV (main results)
    import pandas as pd
    
    df_data = {
        'time': result['frame_times'],
        'speed_ms': result['metric_speeds'],
        'confidence': result['confidences'],
        'omega_x': result['omegas'][:, 0],
        'omega_y': result['omegas'][:, 1],
        'omega_z': result['omegas'][:, 2],
    }
    
    if 'metric_speeds_raw' in result:
        df_data['speed_ms_raw'] = result['metric_speeds_raw']
    
    df = pd.DataFrame(df_data)
    csv_path = output_path / "speed_estimates.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Save calibration info
    calib_info = {
        'scale_factor': result['scale_factor'],
        'calibrated': result['calibrated'],
    }
    
    import json
    with open(output_path / "calibration_info.json", 'w') as f:
        json.dump(calib_info, f, indent=2)
    logger.info(f"Saved calibration info to {output_path / 'calibration_info.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate animal speed from optical flow using analytical motion field equations"
    )
    parser.add_argument(
        "flow_dir",
        type=str,
        help="Directory with optical flow results (e.g., outputs/)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV file with ground-truth speeds for calibration (columns: time, speed)",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="time",
        help="Name of time column in CSV (default: time)",
    )
    parser.add_argument(
        "--speed-col",
        type=str,
        default="speed",
        help="Name of speed column in CSV (default: speed)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Camera focal length in pixels (if not in metadata)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: flow_dir/speed_estimation)",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal smoothing of speed estimates",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Smoothing window size (default: 5 frames)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show comparison plot",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=None,
        help="Time offset to align CSV times with video times (seconds). "
             "If not provided, automatically computed as csv_times[0] - frame_times[0]",
    )
    
    args = parser.parse_args()
    
    # Load flow results
    flow_results = load_flow_results(args.flow_dir)
    
    # Get intrinsics
    metadata = flow_results['metadata']
    if 'intrinsics' in metadata:
        intr_data = metadata['intrinsics']
        intrinsics = CameraIntrinsics(
            fx=intr_data['fx'],
            fy=intr_data['fy'],
            cx=intr_data['cx'],
            cy=intr_data['cy'],
        )
    elif args.focal_length is not None:
        # Use provided focal length
        h, w = flow_results['flows'][0].shape[:2]
        intrinsics = CameraIntrinsics(
            fx=args.focal_length,
            fy=args.focal_length,
            cx=w / 2.0,
            cy=h / 2.0,
        )
    else:
        # Estimate from image size
        h, w = flow_results['flows'][0].shape[:2]
        f = max(h, w)  # Rough estimate
        logger.warning(f"No focal length provided - using estimate: {f:.1f} pixels")
        intrinsics = CameraIntrinsics(
            fx=f,
            fy=f,
            cx=w / 2.0,
            cy=h / 2.0,
        )
    
    logger.info(f"Using intrinsics: f={intrinsics.to_matrix()[0,0]:.1f}px, "
                f"c=({intrinsics.to_matrix()[0,2]:.1f}, {intrinsics.to_matrix()[1,2]:.1f})")
    
    # Estimate speeds
    result = estimate_speeds_from_flow(
        flow_results=flow_results,
        intrinsics=intrinsics,
        csv_path=args.csv,
        csv_time_col=args.time_col,
        csv_speed_col=args.speed_col,
        smooth_output=not args.no_smooth,
        smooth_window=args.smooth_window,
        time_offset=args.time_offset,
    )
    
    # Save results
    if args.output_dir is None:
        output_dir = Path(args.flow_dir) / "speed_estimation"
    else:
        output_dir = args.output_dir
    
    save_speed_results(result, output_dir)
    
    # Plot results
    plot_path = Path(output_dir) / "speed_comparison.png"
    plot_speed_comparison(result, output_path=str(plot_path))
    
    if args.plot:
        plot_speed_comparison(result, output_path=None)
    
    # Print summary
    speeds = result['metric_speeds']
    logger.info("\n" + "="*60)
    logger.info("SPEED ESTIMATION SUMMARY")
    logger.info("="*60)
    if result['calibrated']:
        logger.info(f"Scale factor (k):     {result['scale_factor']:.4f} meters")
        logger.info(f"Mean speed:           {speeds.mean():.3f} m/s")
        logger.info(f"Speed range:          {speeds.min():.3f} - {speeds.max():.3f} m/s")
        logger.info(f"Std deviation:        {speeds.std():.3f} m/s")
    else:
        logger.info("⚠️  NOT CALIBRATED - speeds are in relative units")
        logger.info(f"Mean relative speed:  {speeds.mean():.6f} (1/s)")
        logger.info(f"Provide --csv with ground-truth speeds for metric calibration")
    
    logger.info(f"Mean confidence:      {result['confidences'].mean():.3f}")
    logger.info(f"Frames processed:     {len(speeds)}")
    logger.info(f"Results saved to:     {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

