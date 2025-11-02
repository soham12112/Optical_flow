# Quick Start Guide

Get up and running with underwater flow estimation in 5 minutes!

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Test the Installation

Run with the dummy backend (no RAFT required):

```bash
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./outputs_test \
  --backend dummy \
  --resize_long_edge 480 \
  --stride 2
```

This will:
- ✅ Extract and preprocess frames
- ✅ Compute transmission maps
- ✅ Generate (simple) optical flow
- ✅ Create visualizations

**Expected outputs** in `outputs_test/`:
- `optical_flow/flow_vis.mp4` - Flow visualization
- `diagnostics/transmission.mp4` - Transmission maps
- `diagnostics/weights.mp4` - Attenuation weights

## 3. (Optional) Setup RAFT for Real Optical Flow

For production-quality optical flow:

```bash
# Clone RAFT
cd flows/models
git clone https://github.com/princeton-vl/RAFT.git raft_core
cd ../../

# Download weights (choose one):
# RAFT trained on FlyingThings3D
wget -P flows/models/ https://www.dropbox.com/s/4j4z58wuv8o0mfz/models/raft-things.pth

# Or RAFT trained on Sintel
wget -P flows/models/ https://www.dropbox.com/s/8gt9pao3yw7fbnd/models/raft-sintel.pth
```

Then run with RAFT:

```bash
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./outputs_raft \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --resize_long_edge 960
```

## 4. Try the Python API

```python
from flows import process_video

info = process_video(
    video_path="input_video.mp4",
    out_dir="outputs",
    backend="dummy",  # or "raft"
    use_underwater_weights=True,
    weight_params={"mode": "pow", "gamma": 1.0},
)

print(f"Processed {info['num_flow_pairs']} frame pairs")
```

## 5. Understanding the Outputs

### Optical Flow (`optical_flow/`)
- `flow_vis.mp4`: Color wheel visualization (hue=direction, saturation=magnitude)
- `flow_raw/*.npy`: Raw flow arrays (H×W×2) for further analysis

### Scene Flow (`scene_flow/`)
- `scene_flow_vis.mp4`: 3D motion visualization (X→R, Y→G, Z→B)
- `scene_flow_raw/*.npy`: Raw 3D flow arrays (H×W×3)

### Diagnostics (`diagnostics/`)
- `transmission.mp4`: Medium transmission map (bright=clear, dark=degraded)
- `weights.mp4`: Attenuation weights (bright=trusted, dark=downweighted)
- `flow_legend.png`: Color wheel reference for flow visualization

## 6. Common Use Cases

### Fast Preview (Low Resolution)
```bash
python -m flows.run \
  --video input.mp4 \
  --backend dummy \
  --resize_long_edge 480 \
  --stride 3 \
  --scene_flow_track none
```

### High Quality (Full Pipeline)
```bash
python -m flows.run \
  --video input.mp4 \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --scene_flow_track depthpose \
  --resize_long_edge 960 \
  --transmission_method dcp \
  --gamma 1.5
```

### Batch Processing
```python
from pathlib import Path
from flows import process_video

for video in Path("videos/").glob("*.mp4"):
    process_video(
        video_path=str(video),
        out_dir=f"outputs/{video.stem}",
        backend="raft",
    )
```

## 7. Troubleshooting

**"RAFT not found"**
- Use `--backend dummy` for testing, or
- Install RAFT (see step 3)

**"Slow processing"**
- Use `--resize_long_edge 640` or smaller
- Use `--stride 2` to skip frames
- Use `--transmission_method fast`
- Disable scene flow: `--scene_flow_track none`

**"Out of memory"**
- Reduce `--resize_long_edge`
- Use `--device cpu`
- Process shorter clips

## Next Steps

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔬 Check [example_usage.py](example_usage.py) for API examples
- ⚙️ Tune `--gamma` parameter for your specific water conditions
- 📊 Load `.npy` files in Python/MATLAB for custom analysis

## Support

Having issues? Check:
1. Python version (need 3.8+)
2. PyTorch installation (for GPU: install CUDA-enabled version)
3. All dependencies from requirements.txt
4. Video file is readable by OpenCV

For bugs/questions, open an issue on GitHub.

---

**Happy flow estimation! 🌊**

