# Underwater Optical Flow and Scene Flow Tool

A Python tool for estimating optical flow and scene flow in underwater videos with **attenuation-aware weighting**. Based on the wflow-TartanVO approach, this tool down-weights flow estimates in degraded regions (backscatter/attenuation) to improve motion estimation in challenging underwater conditions.

## Features

- 🌊 **Underwater-optimized**: Preprocessing for low-light, hazy underwater video
- 🎯 **Attenuation-aware weighting**: Emphasizes clearer regions using normalized transmission maps
- 🔄 **Optical flow**: Dense 2D motion field estimation using RAFT
- 📐 **Scene flow**: 3D motion estimation via depth + pose fusion
- 🏊 **Speed estimation**: Analytical speed (m/s) from flow using motion field equations (no depth required!)
- 🎨 **Rich visualizations**: Color-coded flow, transmission maps, weight maps
- ⚙️ **Flexible API**: CLI and Python API for easy integration

## Installation

### 1. Clone and install dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### 2. Setup RAFT (optional, for actual optical flow)

For testing, the tool includes a `dummy` backend. For real optical flow, install RAFT:

```bash
# Clone RAFT into the models directory
cd flows/models
git clone https://github.com/princeton-vl/RAFT.git raft_core
cd ../../

# Download RAFT weights (optional)
# wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models/raft-things.pth -P flows/models/
```

### 3. Verify installation

```bash
python -m flows.run --help
```

## Quick Start

### 1. Generate Optical Flow

```bash
# Basic usage with dummy backend (for testing)
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./outputs \
  --backend dummy

# Full pipeline with RAFT (requires RAFT setup)
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./outputs \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --use_underwater_weights 1 \
  --transmission_method fast \
  --resize_long_edge 960
```

### 2. Estimate Speed (NEW!)

After generating optical flow, estimate animal swimming speed in **meters/second**:

```bash
# Estimate speed with CSV calibration
python estimate_speed.py outputs --csv speed_calibration.csv

# This outputs:
# - Speed time-series (m/s)
# - Comparison plots
# - Confidence scores
# - Camera rotation estimates
```

**What you need:**
- Optical flow results from step 1
- CSV with ground-truth speeds for calibration (see `SPEED_ESTIMATION.md`)

**No depth required!** Uses analytical motion field equations to extract speed from flow patterns.

### Python API

```python
from flows import process_video, load_intrinsics

# Process a video
info = process_video(
    video_path="input_video.mp4",
    out_dir="outputs",
    backend="dummy",  # or "raft" with proper setup
    use_underwater_weights=True,
    weight_params={"mode": "pow", "gamma": 1.0},
    resize_long_edge=960,
)

print(f"Processed {info['num_flow_pairs']} frame pairs")
```

## Output Structure

```
outputs/
├── flow/                      # Raw flow arrays (.npy)
│   └── flow_000000.npy
├── weights/                   # Attenuation-aware weight maps
│   └── weight_000000.npy
├── optical_flow_vis.mp4       # Flow visualization video
├── scene_flow/
│   ├── scene_flow_raw/        # Raw 3D scene flow (.npy)
│   │   └── scene_flow_000000_000001.npy
│   └── scene_flow_vis.mp4     # Scene flow visualization
├── speed_estimation/          # Speed analysis (NEW!)
│   ├── speed_estimates.csv    # Time-series: time, speed (m/s), confidence
│   ├── speed_comparison.png   # Plot: estimated vs ground-truth
│   ├── speed_results.npy      # Full results (numpy)
│   └── calibration_info.json  # Scale factor & metadata
├── diagnostics/
│   ├── transmission.mp4       # Transmission map video
│   ├── weights.mp4            # Weight map video
│   └── flow_legend.png        # Color wheel legend
├── metadata.json              # Video metadata & intrinsics
└── intrinsics.json            # Camera intrinsics used
```

## Key Concepts

### Speed Estimation from Optical Flow (NEW!)

This tool includes an **analytical** approach to estimate animal swimming speed (m/s) from optical flow **without requiring depth information**. 

**How it works:**
1. **Extract rotation**: Solve motion field equations to estimate camera angular velocity
2. **Remove rotation**: Subtract rotational flow to isolate translation
3. **Compute relative speed**: Use radial flow pattern to estimate forward speed (relative units)
4. **Calibrate scale**: Match relative speeds to CSV ground-truth to get metric speeds (m/s)

**Key features:**
- ✅ No depth maps required (works with monocular video)
- ✅ Handles camera rotation automatically
- ✅ Attenuation-aware weighting (uses transmission maps)
- ✅ Robust to noise (median-based estimation)
- ✅ CSV calibration for metric speed

See `SPEED_ESTIMATION.md` for full documentation and usage examples.

### Attenuation-Aware Weighting

In underwater imaging, water attenuates light and causes backscatter, degrading image quality non-uniformly. This tool:

1. **Estimates transmission map** `t(x)`: Measures medium clarity at each pixel (0=completely degraded, 1=clear)
2. **Computes weight map** `w(x) = t(x)^γ`: Converts transmission to flow weights
3. **Applies weighted flow** `F_weighted = w(x) · F(x)`: Down-weights unreliable regions

This follows the **wflow-TartanVO** approach ([arXiv:2407.13159](https://arxiv.org/html/2407.13159v1)), which shows that attenuation-aware weighting improves visual odometry in underwater scenes.

### Transmission Estimation Methods

- **`fast`** (default): Quick approximation using luminance-based heuristics
- **`dcp`**: Dark Channel Prior adapted for underwater (slower but more accurate)

### Scene Flow Pipeline

Scene flow represents 3D motion in the world:

1. Estimate depth at time `t` and `t+1` (using MiDaS)
2. Estimate camera pose `(R, t)` from weighted optical flow
3. Backproject and transform: `S = (R·P + t) - P`

## CLI Options

### Core Arguments

- `--video`: Input video path (required)
- `--out_dir`: Output directory (default: `./outputs`)
- `--backend`: Flow backend: `raft`, `raft-small`, or `dummy` (default: `dummy`)
- `--weights`: Path to RAFT weights (`.pth` file)

### Underwater Weighting

- `--use_underwater_weights`: Enable attenuation weighting (1=yes, 0=no)
- `--transmission_method`: Transmission estimation (`fast` or `dcp`)
- `--weight_mode`: Weight function (`pow` or `linear`)
- `--gamma`: Power exponent for weight function (default: 1.0)

### Preprocessing

- `--preprocess`: Apply underwater preprocessing (1=yes, 0=no)
- `--preprocess_gamma`: Gamma correction (default: 0.8, <1 brightens)
- `--denoise_strength`: Denoising strength (default: 10.0)

### Video Processing

- `--resize_long_edge`: Resize to this long edge (default: 960)
- `--fps_out`: Output FPS (default: use original)
- `--stride`: Frame stride (1=every frame, 2=every other, etc.)

### Scene Flow

- `--scene_flow_track`: Scene flow method (`depthpose` or `none`)
- `--depth_model`: Depth model (`DPT_Large`, `DPT_Hybrid`, `MiDaS_small`)
- `--intrinsics_json`: Camera intrinsics JSON file (auto-estimated if not provided)

### Example: Full Pipeline

```bash
python -m flows.run \
  --video underwater_dive.mp4 \
  --out_dir ./dive_analysis \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --scene_flow_track depthpose \
  --use_underwater_weights 1 \
  --transmission_method dcp \
  --gamma 1.5 \
  --preprocess 1 \
  --preprocess_gamma 0.75 \
  --resize_long_edge 960 \
  --stride 1 \
  --verbose
```

## Camera Intrinsics

For accurate scene flow, provide camera intrinsics as JSON:

```json
{
  "fx": 500.0,
  "fy": 500.0,
  "cx": 320.0,
  "cy": 240.0
}
```

Use with `--intrinsics_json camera.json`. If not provided, intrinsics are estimated (less accurate).

## Architecture

```
flows/
├── core/
│   ├── frame_extraction.py        # Video frame extraction
│   ├── underwater_preproc.py      # Denoising + gamma correction
│   ├── transmission.py            # Transmission map & weighting
│   ├── optical_flow.py            # Flow estimation + weighting
│   ├── scene_flow.py              # Depth + pose -> 3D motion
│   ├── speed_estimation.py        # Speed from flow (NEW!)
│   └── processor.py               # Main processing pipeline
├── models/
│   └── raft_wrapper.py            # RAFT model wrapper
├── utils/
│   ├── intrinsics.py              # Camera intrinsics utilities
│   └── visualization.py           # Flow & scene flow visualization
├── run.py                         # CLI entry point
└── estimate_speed.py              # Speed estimation CLI (NEW!)
```

## Documentation

- **`README.md`** (this file): Overview and quick start
- **`SPEED_ESTIMATION.md`**: Complete guide to speed estimation from optical flow
- **`QUICKSTART.md`**: Step-by-step tutorial
- **`PROJECT_SUMMARY.md`**: Technical architecture and design
- **`INSTALLATION.md`**: Detailed installation instructions

## References

This implementation is based on:

1. **Attenuation-Aware Weighting**: [wflow-TartanVO (arXiv:2407.13159)](https://arxiv.org/html/2407.13159v1)
2. **Optical Flow**: [RAFT (ECCV 2020)](https://arxiv.org/abs/2003.12039)
3. **Motion Field Equations**: Standard pinhole camera motion model
4. **Scene Flow Concepts**: [Motion Modeling in Videos](https://medium.com/@d.d.tananaev/motion-modeling-in-videos-part-3-scene-flow-b63006a2ffb2)
5. **Depth Estimation**: [MiDaS (TPAMI 2021)](https://github.com/isl-org/MiDaS)

## Testing Without RAFT

For quick testing without setting up RAFT:

```bash
python -m flows.run \
  --video test_video.mp4 \
  --backend dummy \
  --scene_flow_track none \
  --use_underwater_weights 1
```

The `dummy` backend generates simple gradient-based flow for testing the pipeline.

## Advanced Usage

### Batch Processing

```python
from flows import process_video
from pathlib import Path

videos = Path("videos").glob("*.mp4")

for video_path in videos:
    output_dir = f"outputs/{video_path.stem}"
    process_video(
        video_path=str(video_path),
        out_dir=output_dir,
        backend="raft",
        model_path="flows/models/raft-things.pth",
    )
```

### Custom Intrinsics

```python
from flows import process_video
from flows.utils.intrinsics import CameraIntrinsics

# Create custom intrinsics
intrinsics = CameraIntrinsics(
    fx=600.0,
    fy=600.0,
    cx=320.0,
    cy=240.0,
)

process_video(
    video_path="video.mp4",
    out_dir="outputs",
    intrinsics=intrinsics,
)
```

### Weight Parameter Tuning

The `gamma` parameter in weight function `w = t^γ` controls emphasis:

- `γ = 0.5`: Gentle weighting (less aggressive filtering)
- `γ = 1.0`: Linear weighting (default)
- `γ = 2.0`: Strong weighting (heavily emphasizes clear regions)

Experiment with different values:

```bash
# Conservative weighting
python -m flows.run --video test.mp4 --gamma 0.7

# Aggressive weighting
python -m flows.run --video test.mp4 --gamma 2.0
```

## Troubleshooting

### "RAFT not found" Error

- Make sure RAFT is cloned to `flows/models/raft_core`
- Or use `--backend dummy` for testing

### CUDA Out of Memory

- Reduce `--resize_long_edge` (try 640 or 480)
- Use `--backend raft-small` for smaller memory footprint
- Use `--device cpu` (slower but uses less memory)

### Slow Processing

- Use `--stride 2` to process every other frame
- Use `--resize_long_edge 640` for faster processing
- Use `--transmission_method fast` instead of `dcp`
- Disable scene flow with `--scene_flow_track none`

## License

This tool is provided for research and educational purposes. Individual components (RAFT, MiDaS) have their own licenses.

## Citation

If you use this tool in your research, please cite the underlying methods:

```bibtex
@article{wflow-tartanvo,
  title={Attenuation-Aware Weighted Optical Flow with Medium Transmission Map for Learning-based Visual Odometry in Underwater terrain},
  author={...},
  journal={arXiv:2407.13159},
  year={2024}
}

@inproceedings{RAFT,
  title={RAFT: Recurrent All-Pairs Field Transforms for Optical Flow},
  author={Teed, Zachary and Deng, Jia},
  booktitle={ECCV},
  year={2020}
}
```

## Contributing

Contributions welcome! Please open issues or pull requests.

## Acknowledgments

- **wflow-TartanVO** authors for the attenuation-aware weighting concept
- **RAFT** authors for the excellent optical flow model
- **MiDaS** authors for robust monocular depth estimation

