# Scene Flow Troubleshooting Guide

This guide helps diagnose and fix scene flow generation issues.

## Quick Diagnosis

Run the diagnostic script on your outputs:

```bash
python diagnose_scene_flow.py outputs/
```

This will check:
- Scene flow magnitudes
- Depth estimation quality
- Pose estimation results
- Configuration issues

## Common Issues & Solutions

### Issue 1: Scene Flow Video is Blank or All Black

**Symptoms:**
- `scene_flow_vis.mp4` exists but shows no motion
- All frames appear black or uniform color

**Causes & Solutions:**

1. **Using Dummy Backend** (most common)
   ```bash
   # Problem: Dummy flow creates poor pose estimates
   # Solution: Use RAFT
   python -m flows.run \
     --video input.mp4 \
     --backend raft \
     --weights flows/models/raft-things.pth \
     --scene_flow_track depthpose
   ```

2. **Camera is Stationary**
   - If camera truly isn't moving, scene flow will be near-zero
   - This is expected behavior - try a video with camera motion

3. **Depth Estimation Failing**
   ```bash
   # Check if PyTorch is installed
   python -c "import torch; print(torch.__version__)"
   
   # If not, install:
   pip install torch torchvision
   ```

### Issue 2: Scene Flow Shows Incorrect Motion

**Symptoms:**
- Scene flow video shows motion but it looks wrong
- Motion doesn't match camera movement

**Diagnosis:**

1. **Check optical flow quality first:**
   ```bash
   # View optical flow video
   open outputs/optical_flow/flow_vis.mp4
   
   # If optical flow looks wrong, scene flow will be wrong too
   ```

2. **Run diagnostics:**
   ```bash
   python diagnose_scene_flow.py outputs/
   ```

**Solutions:**

1. **Improve Optical Flow:**
   ```bash
   # Use RAFT instead of dummy
   --backend raft --weights flows/models/raft-things.pth
   
   # Enable attenuation weighting for underwater
   --use_underwater_weights 1 --gamma 1.5
   ```

2. **Provide Accurate Camera Intrinsics:**
   ```bash
   # Create intrinsics.json:
   {
     "fx": 500.0,
     "fy": 500.0,
     "cx": 320.0,
     "cy": 240.0
   }
   
   # Use it:
   --intrinsics_json intrinsics.json
   ```

3. **Check Depth Quality:**
   ```bash
   # View depth as image
   python -c "
   import numpy as np
   import matplotlib.pyplot as plt
   depth = np.load('outputs/diagnostics/depth_000000.npy')
   plt.imshow(depth, cmap='viridis')
   plt.colorbar()
   plt.savefig('depth_check.png')
   print('Saved depth_check.png')
   "
   ```

### Issue 3: "Scene flow magnitude very small" Warning

**What this means:**
- Pose estimation returned near-identity (no motion detected)
- Scene flow will be mostly zero

**Causes:**

1. **Optical flow too weak:**
   - Dummy backend produces low-quality flow
   - Solution: Use RAFT

2. **Image pair too similar:**
   - Frames are nearly identical
   - Solution: Increase stride: `--stride 2` or `--stride 3`

3. **Essential matrix estimation failed:**
   - Not enough feature correspondences
   - Solution: Check if `--resize_long_edge` is too small

### Issue 4: Depth Estimation Using "Dummy Depth"

**Symptoms:**
- Warning: "Using dummy depth estimation"
- Depth maps look like simple radial gradients

**Solution:**

```bash
# Install PyTorch with CUDA (for GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only:
pip install torch torchvision

# Verify:
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

After installing PyTorch, MiDaS will auto-download on first run.

## Improving Scene Flow Quality

### Best Practices

1. **Use RAFT for Optical Flow:**
   ```bash
   python -m flows.run \
     --video input.mp4 \
     --backend raft \
     --weights flows/models/raft-things.pth
   ```

2. **Enable Underwater Weighting:**
   ```bash
   --use_underwater_weights 1 \
   --gamma 1.5 \
   --transmission_method dcp
   ```

3. **Provide Camera Intrinsics:**
   - Essential for accurate 3D reconstruction
   - See example `intrinsics.json` above

4. **Use Appropriate Resolution:**
   ```bash
   # For quality:
   --resize_long_edge 960
   
   # For speed:
   --resize_long_edge 640
   ```

5. **Adjust Frame Stride:**
   ```bash
   # For subtle motion, use consecutive frames:
   --stride 1
   
   # For more motion, skip frames:
   --stride 2  # or 3
   ```

## Diagnostic Tools

### 1. Scene Flow Diagnostics

```bash
python diagnose_scene_flow.py outputs/

# For specific output directory:
python diagnose_scene_flow.py outputs_vid5/
```

**What it checks:**
- Scene flow magnitudes
- Depth estimation quality
- Pose estimation results
- Configuration settings

### 2. Re-generate Visualization

If scene flow `.npy` files are correct but video looks wrong:

```bash
# Regenerate with magnitude visualization
python reprocess_scene_flow_vis.py --output_dir outputs/ --mode magnitude

# Or with XYZ visualization
python reprocess_scene_flow_vis.py --output_dir outputs/ --mode xyz
```

### 3. Manual Inspection

**Check a single scene flow file:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Load scene flow
sf = np.load('outputs/scene_flow/scene_flow_raw/scene_flow_000000_000001.npy')

# Check shape and values
print(f"Shape: {sf.shape}")
print(f"X range: [{sf[:,:,0].min():.4f}, {sf[:,:,0].max():.4f}]")
print(f"Y range: [{sf[:,:,1].min():.4f}, {sf[:,:,1].max():.4f}]")
print(f"Z range: [{sf[:,:,2].min():.4f}, {sf[:,:,2].max():.4f}]")

# Visualize magnitude
mag = np.linalg.norm(sf, axis=-1)
plt.figure(figsize=(10, 6))
plt.imshow(mag, cmap='jet')
plt.colorbar(label='Scene Flow Magnitude')
plt.title(f'Mean: {mag.mean():.4f}, Max: {mag.max():.4f}')
plt.savefig('scene_flow_magnitude.png')
print('Saved scene_flow_magnitude.png')
```

**Check depth:**

```python
import numpy as np
import matplotlib.pyplot as plt

depth = np.load('outputs/diagnostics/depth_000000.npy')

plt.figure(figsize=(10, 6))
plt.imshow(depth, cmap='viridis')
plt.colorbar(label='Depth (arbitrary units)')
plt.title(f'Depth: mean={depth.mean():.2f}, range=[{depth.min():.2f}, {depth.max():.2f}]')
plt.savefig('depth_map.png')
print('Saved depth_map.png')
```

**Check pose:**

```python
import json
import numpy as np

with open('outputs/diagnostics/pose_000000_000001.json') as f:
    pose = json.load(f)

R = np.array(pose['R'])
t = np.array(pose['t'])

print("Rotation matrix:")
print(R)
print(f"\nTranslation: {t}")
print(f"Translation magnitude: {np.linalg.norm(t):.6f}")

# Check if near identity
is_identity = np.allclose(R, np.eye(3), atol=0.1)
print(f"Near identity: {is_identity}")
```

## Expected Values

### Good Scene Flow
```
Magnitude mean: 0.1 - 10.0
Magnitude max: 0.5 - 50.0
X, Y, Z ranges: Non-zero with reasonable spread
```

### Problem Indicators
```
Magnitude mean: < 0.01  → Camera stationary or pose failed
Magnitude max: < 0.001  → Serious issue, check depth & flow
All values near zero   → Check optical flow quality
```

## Step-by-Step Debugging

### Step 1: Verify Installation
```bash
python test_installation.py
```

### Step 2: Check Configuration
```bash
cat outputs/processing_info.json
```

Look for:
- `backend`: Should be `raft` for quality (not `dummy`)
- `scene_flow_track`: Should be `depthpose`
- `use_underwater_weights`: Consider setting to `true`

### Step 3: Run Diagnostics
```bash
python diagnose_scene_flow.py outputs/
```

### Step 4: Check Optical Flow
```bash
open outputs/optical_flow/flow_vis.mp4
```

If optical flow looks wrong, fix that first!

### Step 5: Check Depth
```bash
ls outputs/diagnostics/depth_*.npy
```

If using dummy depth, install PyTorch.

### Step 6: Reprocess if Needed
```bash
# If only visualization is wrong:
python reprocess_scene_flow_vis.py --output_dir outputs/

# If computation is wrong, reprocess video:
python -m flows.run \
  --video input.mp4 \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --scene_flow_track depthpose \
  --use_underwater_weights 1
```

## Still Having Issues?

### Create a Minimal Test Case

```bash
# Try with a short clip (first 5 seconds):
ffmpeg -i input.mp4 -t 5 test_clip.mp4

# Process it:
python -m flows.run \
  --video test_clip.mp4 \
  --out_dir test_outputs \
  --backend raft \
  --resize_long_edge 640 \
  --verbose

# Check logs and diagnostics:
python diagnose_scene_flow.py test_outputs/
```

### Check Logs

Run with `--verbose` flag and look for:
- "Using dummy depth" warnings
- "Scene flow magnitude very small" warnings
- "Essential matrix estimation failed" warnings
- "Insufficient matches" warnings

### Report Issue

If still stuck, provide:
1. Command used
2. Output of `python diagnose_scene_flow.py outputs/`
3. Configuration from `outputs/processing_info.json`
4. Any warnings/errors from logs
5. Sample frame if possible

## Performance Tips

Scene flow computation is slower than optical flow alone:

```bash
# Fast (5-10 fps):
--resize_long_edge 640 --stride 2

# Quality (2-5 fps):
--resize_long_edge 960 --stride 1

# Use GPU if available - much faster!
--device cuda  # automatic if CUDA available
```

---

**Remember:** Good scene flow requires:
1. ✅ Good optical flow (use RAFT, not dummy)
2. ✅ Good depth (install PyTorch for MiDaS)
3. ✅ Camera intrinsics (provide or use defaults)
4. ✅ Actual camera motion (stationary camera = zero scene flow)

