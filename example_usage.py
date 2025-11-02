"""
Example usage of the underwater flows API.
"""

from flows import process_video
from flows.utils.intrinsics import CameraIntrinsics
from pathlib import Path


def example_basic():
    """Basic usage with dummy backend (no RAFT required)."""
    print("=" * 60)
    print("Example 1: Basic usage with dummy backend")
    print("=" * 60)
    
    # Check if we have a test video
    if not Path("input_video.mp4").exists():
        print("⚠️  No input_video.mp4 found. Skipping example.")
        return
    
    info = process_video(
        video_path="input_video.mp4",
        out_dir="outputs/example_basic",
        backend="dummy",  # No RAFT needed
        use_underwater_weights=True,
        weight_params={"mode": "pow", "gamma": 1.0},
        resize_long_edge=640,  # Small for speed
        stride=2,  # Process every other frame
        scene_flow_track="none",  # Disable scene flow for speed
    )
    
    print(f"\n✓ Processed {info['num_flow_pairs']} frame pairs")
    print(f"✓ Outputs saved to: outputs/example_basic")


def example_with_raft():
    """Full pipeline with RAFT optical flow."""
    print("\n" + "=" * 60)
    print("Example 2: Full pipeline with RAFT")
    print("=" * 60)
    
    # Check if RAFT is available
    if not Path("flows/models/raft_core").exists():
        print("⚠️  RAFT not found. Clone it to flows/models/raft_core")
        print("   git clone https://github.com/princeton-vl/RAFT.git flows/models/raft_core")
        return
    
    if not Path("input_video.mp4").exists():
        print("⚠️  No input_video.mp4 found. Skipping example.")
        return
    
    info = process_video(
        video_path="input_video.mp4",
        out_dir="outputs/example_raft",
        backend="raft",
        model_path="flows/models/raft-things.pth",  # Download separately
        scene_flow_track="depthpose",
        use_underwater_weights=True,
        weight_params={"mode": "pow", "gamma": 1.5},
        transmission_method="fast",
        resize_long_edge=960,
        preprocess_underwater=True,
        gamma=0.8,
    )
    
    print(f"\n✓ Processed {info['num_flow_pairs']} frame pairs")
    print(f"✓ Outputs saved to: outputs/example_raft")


def example_with_intrinsics():
    """Using custom camera intrinsics."""
    print("\n" + "=" * 60)
    print("Example 3: Custom camera intrinsics")
    print("=" * 60)
    
    if not Path("input_video.mp4").exists():
        print("⚠️  No input_video.mp4 found. Skipping example.")
        return
    
    # Define custom intrinsics
    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
    )
    
    print(f"Using intrinsics: {intrinsics}")
    
    info = process_video(
        video_path="input_video.mp4",
        out_dir="outputs/example_intrinsics",
        backend="dummy",
        intrinsics=intrinsics,
        scene_flow_track="depthpose",
        use_underwater_weights=True,
        resize_long_edge=640,
    )
    
    print(f"\n✓ Processed {info['num_flow_pairs']} frame pairs")
    print(f"✓ Scene flow computed with custom intrinsics")


def example_batch_processing():
    """Process multiple videos."""
    print("\n" + "=" * 60)
    print("Example 4: Batch processing")
    print("=" * 60)
    
    # Find all .mp4 files
    videos = list(Path(".").glob("*.mp4"))
    
    if not videos:
        print("⚠️  No .mp4 files found in current directory")
        return
    
    print(f"Found {len(videos)} videos to process")
    
    for video_path in videos[:2]:  # Process first 2 only
        print(f"\nProcessing: {video_path.name}")
        
        output_dir = f"outputs/batch/{video_path.stem}"
        
        info = process_video(
            video_path=str(video_path),
            out_dir=output_dir,
            backend="dummy",
            use_underwater_weights=True,
            resize_long_edge=480,  # Small for speed
            stride=3,  # Every 3rd frame
            scene_flow_track="none",
        )
        
        print(f"  ✓ {info['num_flow_pairs']} pairs -> {output_dir}")


def example_weight_comparison():
    """Compare different weight parameters."""
    print("\n" + "=" * 60)
    print("Example 5: Weight parameter comparison")
    print("=" * 60)
    
    if not Path("input_video.mp4").exists():
        print("⚠️  No input_video.mp4 found. Skipping example.")
        return
    
    gammas = [0.5, 1.0, 2.0]
    
    for gamma in gammas:
        print(f"\nProcessing with gamma={gamma}")
        
        info = process_video(
            video_path="input_video.mp4",
            out_dir=f"outputs/comparison/gamma_{gamma}",
            backend="dummy",
            use_underwater_weights=True,
            weight_params={"mode": "pow", "gamma": gamma},
            resize_long_edge=480,
            stride=2,
            scene_flow_track="none",
        )
        
        print(f"  ✓ Saved to outputs/comparison/gamma_{gamma}")
    
    print("\n💡 Compare the weight maps in diagnostics/ folders to see the difference!")


if __name__ == "__main__":
    # Run examples
    example_basic()
    # example_with_raft()  # Uncomment if you have RAFT setup
    # example_with_intrinsics()
    # example_batch_processing()
    # example_weight_comparison()
    
    print("\n" + "=" * 60)
    print("Examples complete! Check the outputs/ directory.")
    print("=" * 60)

