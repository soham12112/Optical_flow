# Speed Estimation Quick Start Guide

This guide shows you how to get from **optical flow** → **speed in meters/second** using the analytical speed estimation module.

## What You Need

1. ✅ **Optical flow results** (from your existing pipeline)
2. ✅ **CSV file** with ground-truth speeds for calibration
3. ✅ **Camera intrinsics** (focal length) - can be estimated if unknown

## Step-by-Step Workflow

### Step 1: Generate Optical Flow (If Not Already Done)

```bash
# Process your underwater video to get optical flow
python example_usage.py input_video.mp4 --output-dir outputs

# This creates:
# - outputs/flow/*.npy (optical flow fields)
# - outputs/weights/*.npy (attenuation-aware weights)
# - outputs/metadata.json (video info + intrinsics)
```

### Step 2: Prepare Calibration CSV

Create a CSV file with time and speed measurements:

**`my_calibration.csv`:**
```csv
time,speed
0.0,0.5
2.5,1.2
5.0,1.8
7.5,2.1
10.0,1.5
15.0,0.8
20.0,1.3
```

**Requirements:**
- **time**: Time in seconds (aligned with video timeline)
- **speed**: Speed in meters/second
- **Coverage**: Should span the video duration
- **Quantity**: At least 10-20 measurements (more is better)
- **Variety**: Include different speeds (slow, medium, fast)

**How to get calibration data:**
- **Manual tracking**: Measure distance between known landmarks/features
- **GPS/IMU data**: If available on camera/platform
- **Literature values**: Known swimming speeds for the species
- **Reference objects**: Use objects of known size to estimate distance traveled

### Step 3: Run Speed Estimation

```bash
# Basic usage
python estimate_speed.py outputs --csv my_calibration.csv

# With options
python estimate_speed.py outputs \
  --csv my_calibration.csv \
  --focal-length 1200 \
  --smooth-window 7 \
  --plot
```

**Output:**
```
Calibrated scale factor k = 5.2341 meters
Calibration quality: RMSE=0.12 m/s, MAE=0.09 m/s, R²=0.95

Speed Estimation Summary:
  Mean speed:     1.234 m/s
  Speed range:    0.523 - 2.156 m/s
  Mean confidence: 0.847
  
Results saved to: outputs/speed_estimation/
```

### Step 4: Review Results

**Files created:**

1. **`speed_estimates.csv`** - Main results
   ```csv
   time,speed_ms,confidence,omega_x,omega_y,omega_z
   0.000,1.234,0.87,-0.02,0.15,0.03
   0.033,1.245,0.89,-0.01,0.14,0.02
   ...
   ```

2. **`speed_comparison.png`** - Visualization
   - Top: Estimated vs ground-truth speeds
   - Middle: Confidence scores over time
   - Bottom: Camera rotation (angular velocity)

3. **`calibration_info.json`** - Metadata
   ```json
   {
     "scale_factor": 5.2341,
     "calibrated": true
   }
   ```

## Understanding the Results

### Calibration Quality

Check the calibration metrics in the output:

- **RMSE < 0.2 m/s**: ✅ Excellent
- **RMSE 0.2-0.5 m/s**: ✅ Good
- **RMSE > 0.5 m/s**: ⚠️ Review CSV and flow quality

- **R² > 0.85**: ✅ Excellent fit
- **R² 0.7-0.85**: ✅ Good fit  
- **R² < 0.7**: ⚠️ Check for issues

### Confidence Scores

Speed estimates include per-frame confidence (0-1):

- **Confidence > 0.7**: High quality, reliable estimate
- **Confidence 0.4-0.7**: Moderate quality
- **Confidence < 0.4**: Low quality, may want to filter out

Low confidence can indicate:
- Poor visibility (turbid water)
- Heavy backscatter/attenuation
- Extreme camera rotation
- Dynamic objects in scene

### Scale Factor (k)

The scale factor represents the **average scene depth** in meters:

```
k ≈ Z₀ (depth to scene in meters)
```

- Typical values: 3-10 meters for underwater animal tracking
- If k seems unreasonable, check CSV units and time alignment

## Common Issues & Solutions

### ❌ "CSV times do not overlap with frame times"

**Cause:** CSV time range doesn't match video
**Solution:** 
- Check CSV times are in seconds
- Ensure CSV covers video duration
- Verify video starts at t=0

### ❌ High RMSE or poor calibration

**Possible causes:**
1. Insufficient CSV samples (need 10-20+)
2. CSV speeds are incorrect or wrong units
3. Poor optical flow quality
4. Incorrect focal length

**Solutions:**
- Add more calibration points
- Verify CSV speed units (must be m/s)
- Improve flow preprocessing
- Check/provide correct focal length: `--focal-length 1200`

### ❌ Low confidence scores

**Causes:**
- Turbid/low-visibility water
- Extreme camera shake
- Dynamic objects (other animals, bubbles)

**Solutions:**
- Filter low-confidence frames in post-processing
- Increase smoothing: `--smooth-window 9`
- Improve underwater preprocessing

### ❌ Speeds seem too high/low

**Causes:**
- Scale factor calibration error
- Incorrect focal length
- CSV speeds incorrect

**Solutions:**
- Verify CSV speeds are in m/s
- Provide correct focal length explicitly
- Recalibrate with better CSV data

## Advanced Usage

### Test Without Real Data

Test the algorithm with synthetic data:

```bash
python test_speed_estimation.py
```

This runs validation tests with known ground truth and creates visualizations.

### Custom CSV Columns

If your CSV has different column names:

```bash
python estimate_speed.py outputs \
  --csv tracking_data.csv \
  --time-col timestamp_sec \
  --speed-col velocity_mps
```

### Without Calibration

If you don't have CSV calibration data yet:

```bash
# Get relative speeds (arbitrary units)
python estimate_speed.py outputs
```

This outputs relative speeds that you can calibrate later.

### Python API

```python
from flows.core.speed_estimation import AnalyticalSpeedEstimator, load_speed_csv
from flows.utils.intrinsics import CameraIntrinsics
import numpy as np

# Setup
intrinsics = CameraIntrinsics(fx=1200, fy=1200, cx=960, cy=540)
estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)

# Load flow
flow = np.load('outputs/flow/flow_000000.npy')
weight = np.load('outputs/weights/weight_000000.npy')

# Estimate (relative units)
estimate = estimator.estimate_speed(flow, weight, dt=1/30)
print(f"Relative speed: {estimate.rel_speed:.4f}")

# Calibrate
csv_times, csv_speeds = load_speed_csv('calibration.csv')
frame_times = np.arange(0, 10, 1/30)
rel_speeds = np.array([...])  # Your estimates

k = estimator.calibrate_scale(frame_times, rel_speeds, csv_times, csv_speeds)

# Now get metric speeds
estimate = estimator.estimate_speed(flow, weight, dt=1/30)
print(f"Speed: {estimate.speed_ms:.3f} m/s")
```

## How It Works (Technical Summary)

The estimator uses **motion field equations** for a pinhole camera:

1. **Optical flow decomposition:**
   ```
   flow = translational component + rotational component
   ```

2. **Estimate rotation (ω)** from flow using rotational terms
   - Uses robust weighted least squares
   - Weights incorporate attenuation/reliability

3. **Remove rotation** to get pure translation

4. **Extract relative speed** from radial pattern:
   ```
   |flow| ≈ (V / Z₀) * r
   where r = √(x² + y²)
   ```

5. **Calibrate scale** factor k = Z₀ from CSV:
   ```
   speed_metric = k * speed_relative
   ```

**Key advantage:** No depth information required! Works with monocular video.

## Tips for Best Results

### 1. Calibration CSV
- ✅ More samples = better calibration
- ✅ Include variety of speeds
- ✅ Spread measurements across video duration
- ✅ Verify units are m/s

### 2. Camera Setup
- ✅ Provide accurate focal length if possible
- ✅ Stable mounting reduces rotation noise
- ✅ Minimize roll/pitch for best accuracy

### 3. Scene Conditions
- ✅ Clear water → higher confidence
- ✅ Rigid background (seafloor) → better estimates
- ✅ Forward-dominant motion → most accurate
- ❌ Avoid highly dynamic scenes with multiple moving objects

### 4. Post-Processing
- ✅ Use temporal smoothing (enabled by default)
- ✅ Filter low-confidence frames if needed
- ✅ Validate against expected speed ranges

## Next Steps

1. **Generate optical flow** from your video
2. **Prepare calibration CSV** with 10-20 speed measurements
3. **Run speed estimation:** `python estimate_speed.py outputs --csv calibration.csv`
4. **Review results** in `outputs/speed_estimation/`
5. **Iterate if needed:** Adjust CSV, focal length, or smoothing

For more details, see:
- **`SPEED_ESTIMATION.md`**: Complete technical documentation
- **`test_speed_estimation.py`**: Validation tests and examples
- **`estimate_speed.py`**: Full CLI options

## Questions?

- Check `SPEED_ESTIMATION.md` troubleshooting section
- Run tests: `python test_speed_estimation.py`
- Verify optical flow quality first
- Ensure CSV times align with video

Good luck! 🐋

