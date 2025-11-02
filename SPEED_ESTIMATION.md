# Speed Estimation from Optical Flow

This module provides an **analytical** approach to estimate animal swimming speed (m/s) from optical flow without requiring depth information.

## Overview

The system uses **motion field equations** to:
1. Estimate and remove camera rotation from optical flow
2. Extract forward speed from the radial flow pattern
3. Calibrate scale using CSV ground-truth speeds
4. Output metric speed estimates (m/s)

This is **variant 2B** (no-depth) from the analytical framework.

## Quick Start

### 1. Process Your Video to Get Optical Flow

First, generate optical flow from your underwater video:

```bash
python example_usage.py input_video.mp4 --output-dir outputs
```

This creates flow fields, weights (attenuation-aware), and metadata.

### 2. Prepare CSV with Ground-Truth Speeds

Create a CSV file with time and speed measurements for calibration:

**`speed_calibration.csv`:**
```csv
time,speed
0.0,0.5
1.0,1.2
2.0,1.8
3.0,2.1
5.0,2.3
10.0,1.5
...
```

**Requirements:**
- `time`: Time in seconds (should overlap with video time range)
- `speed`: Speed in meters/second
- Need at least 10-20 measurements spanning the video
- Include variety of speeds (slow and fast swimming)
- More measurements = better calibration

**How to get ground-truth speeds:**
- Manual tracking: measure distance traveled between known landmarks
- GPS/IMU data (if available on camera)
- Known swimming speeds from literature for the species
- Reference objects of known size in the scene

### 3. Estimate Speed

Run the speed estimator with your flow data and CSV:

```bash
python estimate_speed.py outputs --csv speed_calibration.csv
```

This will:
- Load optical flow from `outputs/`
- Estimate relative speeds using motion field equations
- Calibrate scale factor using CSV
- Save results to `outputs/speed_estimation/`
- Generate comparison plots

**Output files:**
- `speed_estimates.csv`: Time-series of speed estimates
- `speed_comparison.png`: Visual comparison with ground truth
- `speed_results.npy`: Full results (numpy format)
- `calibration_info.json`: Calibration metadata

## Command-Line Options

```bash
python estimate_speed.py <flow_dir> [OPTIONS]

Required:
  flow_dir              Directory with optical flow results

Options:
  --csv FILE            CSV with ground-truth speeds for calibration
  --time-col NAME       Name of time column (default: time)
  --speed-col NAME      Name of speed column (default: speed)
  --focal-length F      Camera focal length in pixels (optional)
  --output-dir DIR      Output directory (default: flow_dir/speed_estimation)
  --no-smooth           Disable temporal smoothing
  --smooth-window N     Smoothing window size (default: 5 frames)
  --plot                Show interactive plot
```

### Examples

**With focal length and custom CSV columns:**
```bash
python estimate_speed.py outputs \
  --csv my_speeds.csv \
  --time-col timestamp \
  --speed-col velocity_ms \
  --focal-length 1200 \
  --plot
```

**Without calibration (relative speeds only):**
```bash
python estimate_speed.py outputs --output-dir speed_uncalibrated
```

This will output relative speeds (arbitrary units) without metric calibration.

## Understanding the Output

### Speed Estimates CSV

```csv
time,speed_ms,confidence,omega_x,omega_y,omega_z,speed_ms_raw
0.0,1.23,0.87,-0.02,0.15,0.03,1.28
0.033,1.25,0.89,-0.01,0.14,0.02,1.31
...
```

- `time`: Frame time (seconds)
- `speed_ms`: Estimated speed (m/s, smoothed)
- `speed_ms_raw`: Raw speed before smoothing (if smoothing enabled)
- `confidence`: Confidence score (0-1), based on flow consistency
- `omega_x`, `omega_y`, `omega_z`: Angular velocity (rad/s) - pitch, yaw, roll rates

### Calibration Quality Metrics

After calibration, check the output for:

```
Calibration quality: RMSE=0.15 m/s, MAE=0.12 m/s, R²=0.92
```

- **RMSE** (Root Mean Square Error): Average error magnitude
  - < 0.2 m/s: Excellent
  - 0.2-0.5 m/s: Good
  - \> 0.5 m/s: Review CSV and flow quality

- **MAE** (Mean Absolute Error): Average absolute error
  
- **R²**: Correlation (0-1)
  - \> 0.85: Excellent fit
  - 0.7-0.85: Good fit
  - < 0.7: Check for issues

### Confidence Scores

Low confidence (<0.5) may indicate:
- Poor optical flow quality (low visibility, blur)
- Highly turbid water (low transmission)
- Non-rigid scene motion (other animals, moving objects)
- Extreme rotation overwhelming translation

Filter or review low-confidence frames manually.

## Technical Details

### Motion Field Equations

The optical flow at pixel (x, y) in normalized camera coordinates is:

```
ẋ = (-Tx + x*Tz)/Z + x*y*ωx - (1+x²)*ωy + y*ωz
ẏ = (-Ty + y*Tz)/Z + (1+y²)*ωx - x*y*ωy - x*ωz
```

Where:
- `(Tx, Ty, Tz)`: Camera linear velocity (m/s)
- `(ωx, ωy, ωz)`: Camera angular velocity (rad/s)
- `Z`: Scene depth (unknown)

### No-Depth Algorithm

1. **Estimate Rotation**: Solve for `ω = (ωx, ωy, ωz)` using rotational terms
   - Uses robust weighted least squares (Huber loss)
   - Weights incorporate flow reliability and underwater attenuation

2. **Remove Rotation**: Subtract rotational flow from total flow
   - Leaves translational component

3. **Extract Relative Speed**: For forward motion `V`:
   ```
   |flow| ≈ (V / Z₀) * r
   ```
   Where `r = √(x² + y²)` is radial distance.
   
   Estimate: `V_rel = median(|flow| / r)`

4. **Calibrate Scale**: Find scale factor `k = Z₀` such that:
   ```
   speed_metric = k * V_rel
   ```
   Matches CSV ground truth.

### Weights

Pixel weights combine:
- **Attenuation weight**: From underwater transmission map (suppress hazy regions)
- **Flow reliability**: From forward-backward consistency
- **Robust weight**: Downweight outliers (Huber/IRLS)

### Assumptions

This approach assumes:
- Locally dominant planar or slowly varying depth
- Forward-dominant motion (works for typical animal swimming)
- Small-to-moderate rotation (handled via subtraction)
- Rigid background (seafloor, rocks)

It does **not** require:
- Depth maps or 3D reconstruction
- Known scene geometry
- IMU/gyroscope data (though these can improve rotation estimates)

## Troubleshooting

### Issue: High RMSE or Poor Calibration

**Possible causes:**
1. **CSV times don't overlap video**: Ensure CSV time range covers video
2. **Insufficient CSV samples**: Need 10-20+ varied speeds
3. **Poor optical flow**: Check flow visualization for quality
4. **Incorrect focal length**: Provide `--focal-length` explicitly
5. **Non-forward motion**: Algorithm assumes mostly forward swimming

**Solutions:**
- Verify CSV time alignment with video
- Add more calibration points
- Improve flow with better preprocessing (see optical flow docs)
- Use EXIF data or calibration pattern for focal length

### Issue: Low Confidence Scores

**Possible causes:**
1. Low underwater visibility
2. Highly turbid/attenuated regions
3. Camera shake or extreme rotation
4. Dynamic objects in scene (other animals, bubbles)

**Solutions:**
- Focus on higher-confidence segments
- Improve underwater image enhancement (transmission maps)
- Filter frames with confidence < threshold (e.g., 0.4)
- Use temporal smoothing (enabled by default)

### Issue: Speed Estimates Too High/Low

**Possible causes:**
1. Scale factor calibration error
2. Incorrect CSV speeds
3. Focal length incorrect
4. Depth variation (algorithm assumes constant depth)

**Solutions:**
- Verify CSV speed units (must be m/s)
- Check focal length estimate
- Recalibrate with better CSV data
- For depth variation, consider depth-aware variant (2A) - requires depth maps

### Issue: Jittery/Noisy Speeds

**Solutions:**
- Increase smoothing window: `--smooth-window 9`
- Use Gaussian smoothing post-hoc
- Filter low-confidence frames
- Improve optical flow quality (preprocessing)

## Integration with Your Pipeline

### Use in Python Scripts

```python
from flows.core.speed_estimation import (
    AnalyticalSpeedEstimator,
    load_speed_csv,
    smooth_speeds,
)
from flows.utils.intrinsics import CameraIntrinsics

# Create estimator
intrinsics = CameraIntrinsics(fx=1200, fy=1200, cx=960, cy=540)
estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)

# Estimate speed from single flow field
import numpy as np
flow = np.load('outputs/flow/flow_000000.npy')
weight = np.load('outputs/weights/weight_000000.npy')

estimate = estimator.estimate_speed(flow, weight, dt=1/30)
print(f"Relative speed: {estimate.rel_speed:.4f}")
print(f"Omega: {estimate.omega}")
print(f"Confidence: {estimate.confidence:.3f}")

# Calibrate from CSV
csv_times, csv_speeds = load_speed_csv('speed_calibration.csv')
frame_times = np.arange(0, 10, 1/30)
relative_speeds = np.array([...])  # Your estimates

k = estimator.calibrate_scale(frame_times, relative_speeds, csv_times, csv_speeds)
print(f"Scale factor: {k:.4f} meters")

# Now estimate returns metric speeds
estimate = estimator.estimate_speed(flow, weight, dt=1/30)
print(f"Speed: {estimate.speed_ms:.3f} m/s")
```

### Batch Processing

```python
# Process all flows in directory
import glob

flows = sorted(glob.glob('outputs/flow/flow_*.npy'))
weights = sorted(glob.glob('outputs/weights/weight_*.npy'))

speeds = []
for flow_file, weight_file in zip(flows, weights):
    flow = np.load(flow_file)
    weight = np.load(weight_file)
    estimate = estimator.estimate_speed(flow, weight, dt=1/30)
    speeds.append(estimate.speed_ms)

speeds = np.array(speeds)
speeds_smooth = smooth_speeds(speeds, window_size=7, method='savgol')
```

## Advanced Usage

### Custom CSV Format

If your CSV has different column names or structure:

```bash
python estimate_speed.py outputs \
  --csv tracking_data.csv \
  --time-col frame_time_sec \
  --speed-col animal_velocity_mps
```

### Multiple Calibration Segments

For long videos with varying conditions, calibrate on representative segments:

```python
# Calibrate on high-quality segment (e.g., 30-60 seconds)
segment_mask = (frame_times >= 30) & (frame_times <= 60)
k = estimator.calibrate_scale(
    frame_times[segment_mask],
    relative_speeds[segment_mask],
    csv_times,
    csv_speeds,
)
```

### Adaptive Scale Factor

For very long videos or changing depth, recalibrate periodically:

```python
# Sliding window calibration
window_size = 300  # frames
step = 100

for i in range(0, len(frame_times) - window_size, step):
    window = slice(i, i + window_size)
    k_local = estimator.calibrate_scale(
        frame_times[window],
        relative_speeds[window],
        csv_times,
        csv_speeds,
    )
    # Apply k_local to this segment...
```

### Depth-Aware Variant (Future)

If you have depth maps (from MiDaS, ZoeDepth, or stereo), use variant 2A for improved accuracy. This requires modifying the estimator to accept depth inputs. See technical documentation for equations.

## Citation

If you use this speed estimation approach in your research, please cite the motion field equations framework and the optical flow method used.

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify CSV format and time alignment
3. Review optical flow quality (visualize flow fields)
4. Check camera intrinsics (focal length)

See also:
- `QUICKSTART.md` for optical flow generation
- `PROJECT_SUMMARY.md` for overall pipeline documentation

