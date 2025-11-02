#!/usr/bin/env python3
"""
Re-generate scene flow visualization from existing .npy files with improved settings.
Use this if scene flow was already computed but visualization looks wrong.
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import sys

# Import visualization functions
sys.path.insert(0, str(Path(__file__).parent))
from flows.utils.visualization import scene_flow_to_color, save_video_from_frames


def reprocess_scene_flow_visualization(
    output_dir: str = "outputs",
    mode: str = "magnitude",
    fps: float = 24.0,
):
    """
    Re-generate scene flow visualization from existing raw files.
    
    Args:
        output_dir: Directory containing scene flow outputs
        mode: 'magnitude' or 'xyz'
        fps: Output video FPS
    """
    
    output_path = Path(output_dir)
    scene_flow_dir = output_path / "scene_flow" / "scene_flow_raw"
    
    if not scene_flow_dir.exists():
        print(f"❌ Scene flow directory not found: {scene_flow_dir}")
        return
    
    # Get all scene flow files
    sf_files = sorted(scene_flow_dir.glob("scene_flow_*.npy"))
    
    if not sf_files:
        print(f"❌ No scene flow files found in {scene_flow_dir}")
        return
    
    print(f"Found {len(sf_files)} scene flow files")
    print(f"Mode: {mode}")
    print(f"Re-generating visualization...")
    
    frames = []
    
    for sf_file in tqdm(sf_files, desc="Processing"):
        # Load scene flow
        scene_flow = np.load(sf_file)
        
        # Visualize with improved settings
        vis = scene_flow_to_color(scene_flow, mode=mode, percentile=95.0)
        
        # Add magnitude text overlay
        mag = np.linalg.norm(scene_flow, axis=-1)
        mean_mag = mag.mean()
        max_mag = mag.max()
        
        # Add text to frame
        text = f"Mean: {mean_mag:.4f}  Max: {max_mag:.4f}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        frames.append(vis)
    
    # Save video
    output_video = output_path / "scene_flow" / f"scene_flow_vis_{mode}.mp4"
    
    print(f"\nSaving video to: {output_video}")
    save_video_from_frames(frames, str(output_video), fps=fps)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    
    # Load first file for detailed stats
    first_sf = np.load(sf_files[0])
    mag = np.linalg.norm(first_sf, axis=-1)
    
    print(f"Scene flow shape: {first_sf.shape}")
    print(f"Magnitude range: [{mag.min():.4f}, {mag.max():.4f}]")
    print(f"Magnitude mean: {mag.mean():.4f}")
    print(f"Magnitude std: {mag.std():.4f}")
    
    if mag.mean() < 0.01:
        print("\n⚠️  WARNING: Very small magnitudes detected")
        print("   This could indicate:")
        print("   1. Camera is stationary")
        print("   2. Using dummy optical flow backend")
        print("   3. Depth estimation issues")
        print("\n   Try running diagnose_scene_flow.py for more details")
    
    print(f"\n✓ Done! Check: {output_video}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-generate scene flow visualization")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--mode", default="magnitude", choices=["magnitude", "xyz"], 
                       help="Visualization mode")
    parser.add_argument("--fps", type=float, default=24.0, help="Output FPS")
    
    args = parser.parse_args()
    
    reprocess_scene_flow_visualization(args.output_dir, args.mode, args.fps)

