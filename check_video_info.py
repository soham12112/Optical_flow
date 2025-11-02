#!/usr/bin/env python3
"""Check actual video properties."""

import cv2
import sys

def check_video(video_path):
    """Check actual video properties."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print("="*60)
    print("VIDEO PROPERTIES")
    print("="*60)
    print(f"File: {video_path}")
    print(f"\nResolution: {width} x {height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frame count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("="*60)
    
    # Actually count frames to verify
    print("\nVerifying by counting frames...")
    actual_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_count += 1
    
    print(f"Actual frame count (by reading): {actual_count}")
    if actual_count != frame_count:
        print(f"⚠️  Metadata says {frame_count} but actually has {actual_count} frames!")
        actual_duration = actual_count / fps if fps > 0 else 0
        print(f"   Actual duration: {actual_duration:.2f} seconds ({actual_duration/60:.2f} minutes)")
    else:
        print("✓ Frame count matches metadata")
    
    cap.release()
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_video_info.py <video_file>")
        sys.exit(1)
    
    check_video(sys.argv[1])

