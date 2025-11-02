# Quick Reference Card

## 🚀 Getting Started (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test it works
python test_installation.py

# 3. Run on your video
python -m flows.run --video input_video.mp4 --backend dummy --out_dir outputs
```

## 📋 Common Commands

### Basic Processing
```bash
# Fastest (testing only)
python -m flows.run --video IN.mp4 --backend dummy --resize_long_edge 480 --stride 3

# Production (requires RAFT setup)
python -m flows.run --video IN.mp4 --backend raft --weights flows/models/raft-things.pth
```

### With Underwater Weighting
```bash
python -m flows.run \
  --video IN.mp4 \
  --backend raft \
  --use_underwater_weights 1 \
  --transmission_method fast \
  --gamma 1.5
```

### Full Pipeline (Flow + Scene Flow)
```bash
python -m flows.run \
  --video IN.mp4 \
  --backend raft \
  --scene_flow_track depthpose \
  --use_underwater_weights 1 \
  --gamma 1.5 \
  --resize_long_edge 960
```

## 🐍 Python API

```python
from flows import process_video

# Simple
process_video("video.mp4", out_dir="outputs", backend="dummy")

# Advanced
process_video(
    video_path="video.mp4",
    out_dir="outputs",
    backend="raft",
    model_path="flows/models/raft-things.pth",
    use_underwater_weights=True,
    weight_params={"mode": "pow", "gamma": 1.5},
    scene_flow_track="depthpose",
)
```

## 📁 Output Files

```
outputs/
├── optical_flow/flow_vis.mp4      # Flow visualization
├── optical_flow/flow_raw/*.npy    # Raw flow arrays
├── scene_flow/scene_flow_vis.mp4  # 3D motion visualization
├── diagnostics/transmission.mp4   # Transmission maps
└── diagnostics/weights.mp4        # Attenuation weights
```

## ⚙️ Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `--backend` | `dummy`, `raft`, `raft-small` | Flow estimation method |
| `--use_underwater_weights` | `0`, `1` | Enable attenuation weighting |
| `--gamma` | `0.5`-`2.0` (default `1.0`) | Weight emphasis (higher = more aggressive) |
| `--transmission_method` | `fast`, `dcp` | Transmission estimation (fast is faster) |
| `--resize_long_edge` | `480`, `640`, `960` | Processing resolution |
| `--stride` | `1`, `2`, `3` | Frame skip (higher = faster) |
| `--scene_flow_track` | `depthpose`, `none` | 3D motion estimation |

## 🔧 Tuning Guide

### For Speed
```bash
--backend dummy \
--resize_long_edge 480 \
--stride 2 \
--transmission_method fast \
--scene_flow_track none
```

### For Quality
```bash
--backend raft \
--resize_long_edge 960 \
--stride 1 \
--transmission_method dcp \
--scene_flow_track depthpose \
--gamma 1.5
```

### For Hazy Water
```bash
--use_underwater_weights 1 \
--transmission_method dcp \
--gamma 2.0 \          # Aggressive weighting
--preprocess 1 \
--preprocess_gamma 0.75
```

### For Clear Water
```bash
--use_underwater_weights 0 \  # or gamma 0.5
--preprocess 0
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'cv2'" | `pip install opencv-python opencv-contrib-python` |
| "RAFT not found" | Use `--backend dummy` or setup RAFT (see INSTALLATION.md) |
| Out of memory | `--resize_long_edge 640 --device cpu` |
| Too slow | `--stride 2 --resize_long_edge 480 --transmission_method fast` |
| No GPU detected | Install CUDA-enabled PyTorch: see pytorch.org |

## 📖 Documentation Map

- **QUICKSTART.md** → First-time setup & basic usage
- **INSTALLATION.md** → Detailed installation instructions
- **README.md** → Complete feature documentation
- **PROJECT_SUMMARY.md** → Technical overview & architecture
- **example_usage.py** → Python API examples
- **This file** → Quick reference

## 🔍 Reading Outputs

### Load Flow in Python
```python
import numpy as np
flow = np.load("outputs/optical_flow/flow_raw/flow_000000_000001.npy")
print(f"Shape: {flow.shape}")  # (H, W, 2)
u, v = flow[:, :, 0], flow[:, :, 1]  # Horizontal, vertical
```

### Load Scene Flow
```python
scene_flow = np.load("outputs/scene_flow/scene_flow_raw/scene_flow_000000_000001.npy")
X, Y, Z = scene_flow[:, :, 0], scene_flow[:, :, 1], scene_flow[:, :, 2]
magnitude = np.linalg.norm(scene_flow, axis=-1)
```

### Load Weights
```python
transmission = np.load("outputs/diagnostics/transmission_000000.npy")  # [0, 1]
weight = np.load("outputs/diagnostics/weight_000000.npy")  # [0, 1]
```

## 🎯 Examples by Use Case

### Marine Biology (Animal Tracking)
```bash
python -m flows.run \
  --video dolphin.mp4 \
  --backend raft \
  --use_underwater_weights 1 \
  --gamma 1.2 \
  --resize_long_edge 960
```

### Underwater Robotics (Visual Odometry)
```bash
python -m flows.run \
  --video auv_mission.mp4 \
  --backend raft \
  --scene_flow_track depthpose \
  --use_underwater_weights 1 \
  --gamma 1.5 \
  --intrinsics_json camera.json
```

### Video Enhancement Analysis
```bash
python -m flows.run \
  --video raw_footage.mp4 \
  --backend dummy \
  --use_underwater_weights 1 \
  --transmission_method dcp \
  --scene_flow_track none
# Check diagnostics/transmission.mp4 for water quality
```

## ⏱️ Typical Processing Times

| Video Length | Resolution | Backend | Time (RTX 3090) |
|-------------|-----------|---------|-----------------|
| 1 min (30fps) | 480p | Dummy | ~1 min |
| 1 min (30fps) | 960p | RAFT | ~3 min |
| 1 min (30fps) | 960p | RAFT+Depth | ~6 min |

*Without GPU: 5-10x slower*

## 📞 Getting Help

1. Run: `python test_installation.py`
2. Check: Troubleshooting table above
3. Read: INSTALLATION.md troubleshooting section
4. Open GitHub issue with:
   - Command used
   - Full error message
   - `python --version` output
   - `python -c "import torch; print(torch.__version__)"`

---

**Keep this file handy!** Bookmark it for quick reference.

