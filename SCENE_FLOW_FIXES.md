# Scene Flow Fixes - Summary

## What Was Fixed

I've identified and fixed several issues that can cause scene flow to not generate correctly:

### 1. **Improved Depth Estimation Fallback**
- **Problem**: Dummy depth used simple radial pattern (unrealistic)
- **Fix**: Now uses image intensity + spatial variation for better approximation
- **File**: `flows/core/scene_flow.py` - `_dummy_depth()` method

### 2. **Enhanced Scene Flow Visualization**
- **Problem**: Small magnitudes were invisible (all black frames)
- **Fix**: 
  - Changed default mode to "magnitude" (better than XYZ for small values)
  - Added automatic scaling and contrast enhancement
  - Handle near-zero values gracefully
- **File**: `flows/utils/visualization.py` - `scene_flow_to_color()`

### 3. **Added Diagnostic Logging**
- **Problem**: No visibility into what's going wrong
- **Fix**: Added detailed logging for:
  - Scene flow magnitude statistics
  - Warnings for near-zero motion
  - Depth estimation method used
- **Files**: `flows/core/scene_flow.py`, `flows/core/processor.py`

### 4. **Error Handling in Pipeline**
- **Problem**: One failed frame could break entire video
- **Fix**: Try-catch blocks maintain video sync even if scene flow fails
- **File**: `flows/core/processor.py`

### 5. **Added Pose Debugging**
- **Problem**: Hard to tell if pose estimation is working
- **Fix**: Save pose matrices as JSON files for inspection
- **File**: `flows/core/processor.py`

## New Diagnostic Tools

### 1. **Scene Flow Diagnostics Script**
```bash
python diagnose_scene_flow.py outputs/
```

**Checks:**
- ✅ Scene flow file structure
- ✅ Magnitude ranges and statistics
- ✅ Depth quality indicators
- ✅ Pose estimation results
- ✅ Configuration issues
- ✅ Provides specific recommendations

**File**: `diagnose_scene_flow.py`

### 2. **Re-visualization Script**
```bash
python reprocess_scene_flow_vis.py --output_dir outputs/ --mode magnitude
```

**Use when:**
- Scene flow `.npy` files exist but video looks wrong
- Want to try different visualization modes
- Need magnitude overlay on frames

**File**: `reprocess_scene_flow_vis.py`

### 3. **Comprehensive Troubleshooting Guide**
- Step-by-step debugging procedures
- Common issues and solutions
- Expected value ranges
- Manual inspection examples

**File**: `SCENE_FLOW_TROUBLESHOOTING.md`

## How to Use the Fixes

### Option 1: Run Diagnostics on Existing Outputs

If you already processed a video and scene flow looks wrong:

```bash
# 1. Check what's wrong
python diagnose_scene_flow.py outputs/

# 2. Try improved visualization
python reprocess_scene_flow_vis.py --output_dir outputs/ --mode magnitude

# 3. Check the new video
open outputs/scene_flow/scene_flow_vis_magnitude.mp4
```

### Option 2: Reprocess with Better Settings

For best results, reprocess with these settings:

```bash
python -m flows.run \
  --video input_video.mp4 \
  --out_dir outputs_improved \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --scene_flow_track depthpose \
  --use_underwater_weights 1 \
  --gamma 1.5 \
  --resize_long_edge 960 \
  --verbose
```

**Key improvements:**
- `--backend raft`: Better optical flow → better pose estimation
- `--use_underwater_weights 1`: Emphasize clear regions
- `--verbose`: See detailed diagnostic messages

## What to Expect Now

### Improved Diagnostics

When you run processing, you'll now see:

```
Scene flow stats: magnitude mean=0.2341, max=1.5234, std=0.1823
```

If something is wrong, you'll get warnings:

```
⚠️  WARNING: Scene flow magnitude very small - this may indicate:
  1. Camera is nearly stationary
  2. Pose estimation failed (check optical flow quality)
  3. Depth estimation is incorrect
```

### Better Visualizations

Scene flow videos now:
- Use magnitude-based coloring (more visible)
- Auto-scale for contrast
- Include magnitude text overlay (in reprocessed versions)
- Handle near-zero motion gracefully (no more black frames)

### Saved Diagnostic Data

New files in `outputs/diagnostics/`:
- `pose_XXXXXX_XXXXXX.json`: Camera pose matrices
- Better warnings in processing logs

## Most Common Issues & Quick Fixes

### Issue: "All black scene flow video"

**Quick Fix:**
```bash
# Check if using dummy backend
cat outputs/processing_info.json | grep backend

# If "backend": "dummy", reprocess with RAFT:
python -m flows.run --video input.mp4 --backend raft --weights flows/models/raft-things.pth
```

### Issue: "Using dummy depth" warning

**Quick Fix:**
```bash
# Install PyTorch for MiDaS depth
pip install torch torchvision

# Reprocess - MiDaS will auto-download
python -m flows.run --video input.mp4 --backend raft
```

### Issue: "Very small magnitude" warning

**Possible causes:**
1. Camera truly is stationary (expected)
2. Using dummy optical flow (switch to RAFT)
3. Frame stride too small (try `--stride 2`)

**Quick Check:**
```bash
# View optical flow to see if camera is moving
open outputs/optical_flow/flow_vis.mp4

# If flow shows motion but scene flow doesn't, check depth:
python diagnose_scene_flow.py outputs/
```

## Testing the Fixes

### Test 1: Run Diagnostics

```bash
python diagnose_scene_flow.py outputs/
```

Expected: Detailed report with statistics and recommendations

### Test 2: Reprocess Visualization

```bash
python reprocess_scene_flow_vis.py --output_dir outputs/
```

Expected: New video with magnitude text overlay

### Test 3: Full Pipeline

```bash
# Short test video
python -m flows.run \
  --video input_video.mp4 \
  --out_dir test_fix \
  --backend dummy \
  --resize_long_edge 480 \
  --stride 2 \
  --verbose
```

Expected: 
- Detailed logging during processing
- Warning messages if issues detected
- Scene flow video generated (even if magnitudes are small)

## Files Changed

### Core Fixes
- `flows/core/scene_flow.py`: Improved dummy depth, added diagnostics
- `flows/utils/visualization.py`: Better scene flow visualization
- `flows/core/processor.py`: Error handling, pose saving

### New Tools
- `diagnose_scene_flow.py`: Diagnostic script
- `reprocess_scene_flow_vis.py`: Re-visualization script
- `SCENE_FLOW_TROUBLESHOOTING.md`: Comprehensive guide

### Documentation
- `SCENE_FLOW_FIXES.md`: This file
- `QUICK_REFERENCE.md`: Already includes scene flow tips

## Before vs After

### Before
- ❌ Black scene flow videos with no explanation
- ❌ No diagnostic tools
- ❌ Poor dummy depth approximation
- ❌ No visibility into pose estimation
- ❌ Hard to debug issues

### After
- ✅ Clear diagnostic messages during processing
- ✅ Automatic detection of common issues
- ✅ Better visualization (magnitude mode default)
- ✅ Diagnostic script for troubleshooting
- ✅ Re-visualization without reprocessing
- ✅ Improved dummy depth (when PyTorch unavailable)
- ✅ Saved pose data for inspection
- ✅ Comprehensive troubleshooting guide

## Next Steps

1. **If you have existing outputs that look wrong:**
   ```bash
   python diagnose_scene_flow.py outputs/
   python reprocess_scene_flow_vis.py --output_dir outputs/
   ```

2. **If reprocessing:**
   ```bash
   python -m flows.run --video input.mp4 --backend raft --verbose
   ```

3. **If still having issues:**
   - Read `SCENE_FLOW_TROUBLESHOOTING.md`
   - Check optical flow quality first
   - Ensure PyTorch is installed for MiDaS
   - Provide camera intrinsics if available

## Questions?

- Read: `SCENE_FLOW_TROUBLESHOOTING.md` for detailed debugging
- Run: `python diagnose_scene_flow.py outputs/` for automatic diagnosis
- Check: Logs with `--verbose` flag for detailed messages

---

**Summary**: Scene flow should now work much better, with clear diagnostics when it doesn't!

