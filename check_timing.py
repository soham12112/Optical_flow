#!/usr/bin/env python3
"""
Quick diagnostic to check video and CSV timing alignment.
"""

import pandas as pd
import numpy as np
import sys

def parse_time(time_str):
    """Convert time string to seconds."""
    time_str = str(time_str).strip()
    
    # Try direct numeric conversion first
    try:
        return float(time_str)
    except ValueError:
        pass
    
    # Try MM:SS.S or HH:MM:SS.S format
    try:
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS.S
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        elif len(parts) == 3:  # HH:MM:SS.S
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        pass
    
    return np.nan

def check_timing(csv_path, video_frames, video_fps):
    """Check timing alignment between CSV and video."""
    
    print("="*60)
    print("TIMING DIAGNOSTIC")
    print("="*60)
    
    # Video info
    video_duration = video_frames / video_fps
    print(f"\nVIDEO:")
    print(f"  Frames:     {video_frames}")
    print(f"  FPS:        {video_fps}")
    print(f"  Duration:   {video_duration:.2f} seconds ({video_duration/60:.2f} minutes)")
    print(f"  Time range: 0.00 - {video_duration:.2f} seconds")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    times = df['time'].apply(parse_time).values
    speeds = pd.to_numeric(df['speed'], errors='coerce').values
    
    # Remove NaN
    valid_mask = ~(np.isnan(times) | np.isnan(speeds))
    times = times[valid_mask]
    speeds = speeds[valid_mask]
    
    csv_duration = times[-1] - times[0]
    
    print(f"\nCSV:")
    print(f"  Entries:    {len(times)}")
    print(f"  Time range (absolute): {times[0]:.2f} - {times[-1]:.2f} seconds")
    print(f"  Duration:   {csv_duration:.2f} seconds ({csv_duration/60:.2f} minutes)")
    print(f"  Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    
    # Show first and last few entries
    print(f"\n  First 3 times: {times[0]:.2f}, {times[1]:.2f}, {times[2]:.2f}")
    print(f"  Last 3 times:  {times[-3]:.2f}, {times[-2]:.2f}, {times[-1]:.2f}")
    
    # Alignment analysis
    print(f"\nALIGNMENT:")
    auto_offset = times[0]  # Assuming video starts at 0
    print(f"  Auto offset: {auto_offset:.2f} seconds")
    print(f"  After alignment:")
    print(f"    CSV time range: 0.00 - {csv_duration:.2f} seconds")
    print(f"    Video time range: 0.00 - {video_duration:.2f} seconds")
    
    # Check overlap
    if csv_duration < video_duration:
        print(f"\n  ⚠️  WARNING: CSV duration ({csv_duration:.1f}s) < video duration ({video_duration:.1f}s)")
        print(f"      Missing {video_duration - csv_duration:.1f} seconds of CSV data")
        print(f"      Calibration will only use first {csv_duration:.1f} seconds of video")
    elif csv_duration > video_duration:
        print(f"\n  ⚠️  WARNING: CSV duration ({csv_duration:.1f}s) > video duration ({video_duration:.1f}s)")
        print(f"      CSV has {csv_duration - video_duration:.1f} seconds of extra data")
        print(f"      Will only use CSV data matching video duration")
    else:
        print(f"\n  ✓ Good match! CSV and video have same duration")
    
    # Show time conversion example
    print(f"\nTIME CONVERSION EXAMPLE:")
    print(f"  CSV first entry: {df['time'].iloc[0]} → {times[0]:.2f} seconds (absolute)")
    print(f"  After alignment: {times[0] - auto_offset:.2f} seconds (video time)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_timing.py <csv_file> [frames] [fps]")
        print("Example: python check_timing.py vid5_1min.csv 2352 30")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 2352
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    
    check_timing(csv_path, frames, fps)

