#!/usr/bin/env python3
"""
Diagnostic script for scene flow issues.
Analyzes scene flow outputs to identify problems.
"""

import numpy as np
import json
from pathlib import Path
import sys


def diagnose_scene_flow(output_dir: str = "outputs"):
    """Diagnose scene flow outputs."""
    
    output_path = Path(output_dir)
    
    print("=" * 60)
    print(f"Scene Flow Diagnostics for: {output_dir}")
    print("=" * 60)
    
    # Check if scene flow directory exists
    scene_flow_dir = output_path / "scene_flow" / "scene_flow_raw"
    if not scene_flow_dir.exists():
        print("❌ No scene flow directory found!")
        print(f"   Expected: {scene_flow_dir}")
        return
    
    # Get scene flow files
    sf_files = sorted(scene_flow_dir.glob("*.npy"))
    if not sf_files:
        print("❌ No scene flow files found!")
        return
    
    print(f"✓ Found {len(sf_files)} scene flow files")
    print()
    
    # Analyze first few files
    print("Analyzing scene flow samples...")
    print("-" * 60)
    
    all_magnitudes = []
    
    for i, sf_file in enumerate(sf_files[:min(10, len(sf_files))]):
        scene_flow = np.load(sf_file)
        
        # Compute statistics
        mag = np.linalg.norm(scene_flow, axis=-1)
        all_magnitudes.append(mag.mean())
        
        print(f"\nFile {i+1}: {sf_file.name}")
        print(f"  Shape: {scene_flow.shape}")
        print(f"  X range: [{scene_flow[..., 0].min():.4f}, {scene_flow[..., 0].max():.4f}]")
        print(f"  Y range: [{scene_flow[..., 1].min():.4f}, {scene_flow[..., 1].max():.4f}]")
        print(f"  Z range: [{scene_flow[..., 2].min():.4f}, {scene_flow[..., 2].max():.4f}]")
        print(f"  Magnitude: mean={mag.mean():.4f}, max={mag.max():.4f}, std={mag.std():.4f}")
        
        # Check for issues
        if mag.mean() < 0.01:
            print("  ⚠️  WARNING: Very small magnitude - likely stationary camera or failed pose")
        if mag.max() < 0.001:
            print("  ❌ ERROR: Near-zero motion detected")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("Overall Statistics")
    print("=" * 60)
    print(f"Average magnitude across all samples: {np.mean(all_magnitudes):.4f}")
    print(f"Min/Max magnitude: [{np.min(all_magnitudes):.4f}, {np.max(all_magnitudes):.4f}]")
    
    # Check depth files
    print("\n" + "=" * 60)
    print("Depth Estimation Check")
    print("=" * 60)
    
    diagnostics_dir = output_path / "diagnostics"
    depth_files = list(diagnostics_dir.glob("depth_*.npy"))
    
    if depth_files:
        print(f"✓ Found {len(depth_files)} depth files")
        
        # Analyze one depth file
        depth = np.load(depth_files[0])
        print(f"\nDepth sample: {depth_files[0].name}")
        print(f"  Shape: {depth.shape}")
        print(f"  Range: [{depth.min():.4f}, {depth.max():.4f}]")
        print(f"  Mean: {depth.mean():.4f}")
        print(f"  Std: {depth.std():.4f}")
        
        # Check if depth looks like dummy depth
        if depth.std() < 0.5:
            print("  ⚠️  WARNING: Low depth variation - may be using dummy depth")
            print("     Install PyTorch and MiDaS for better depth estimation")
    else:
        print("❌ No depth files found")
    
    # Check pose files
    print("\n" + "=" * 60)
    print("Pose Estimation Check")
    print("=" * 60)
    
    pose_files = sorted(diagnostics_dir.glob("pose_*.json"))
    
    if pose_files:
        print(f"✓ Found {len(pose_files)} pose files")
        
        identity_count = 0
        
        for pose_file in pose_files[:5]:
            with open(pose_file) as f:
                pose = json.load(f)
            
            R = np.array(pose['R'])
            t = np.array(pose['t']).flatten()
            
            # Check if R is close to identity
            is_identity = np.allclose(R, np.eye(3), atol=0.1)
            t_norm = np.linalg.norm(t)
            
            if is_identity:
                identity_count += 1
        
        if identity_count > len(pose_files[:5]) * 0.8:
            print(f"  ⚠️  WARNING: {identity_count}/{len(pose_files[:5])} poses are near-identity")
            print("     This indicates:")
            print("       1. Camera is stationary")
            print("       2. Optical flow has very low magnitude")
            print("       3. Pose estimation is failing")
    else:
        print("❌ No pose files found")
    
    # Check processing info
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)
    
    info_file = output_path / "processing_info.json"
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        
        config = info.get('config', {})
        print(f"Backend: {config.get('backend', 'unknown')}")
        print(f"Scene flow track: {config.get('scene_flow_track', 'unknown')}")
        print(f"Underwater weights: {config.get('use_underwater_weights', 'unknown')}")
        
        if config.get('backend') == 'dummy':
            print("\n⚠️  NOTE: Using dummy optical flow backend")
            print("   For better scene flow, use RAFT backend:")
            print("   --backend raft --weights flows/models/raft-things.pth")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    avg_mag = np.mean(all_magnitudes)
    
    if avg_mag < 0.01:
        print("❌ ISSUE: Scene flow magnitude is very small")
        print("\nPossible causes:")
        print("  1. Using dummy backend → Switch to RAFT:")
        print("     python -m flows.run --video INPUT.mp4 --backend raft")
        print("  2. Depth estimation failing → Install PyTorch:")
        print("     pip install torch torchvision")
        print("  3. Camera is truly stationary → Expected behavior")
        print("  4. Optical flow quality is poor → Check flow_vis.mp4")
    elif avg_mag < 0.1:
        print("⚠️  WARNING: Scene flow magnitude is small")
        print("  This may be correct if camera motion is minimal")
        print("  Or consider using RAFT backend for better flow quality")
    else:
        print("✓ Scene flow magnitudes look reasonable!")
        print(f"  Average magnitude: {avg_mag:.4f}")
    
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("1. Check scene_flow_vis.mp4 for visual results")
    print("2. Compare with optical_flow/flow_vis.mp4")
    print("3. Check diagnostics/depth_*.npy for depth quality")
    print("4. If using dummy backend, switch to RAFT for better results")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "outputs"
    
    diagnose_scene_flow(output_dir)

