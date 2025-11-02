# Installation Guide

Complete installation guide for the underwater flows tool.

## Prerequisites

- **Python 3.8+** (3.10 recommended)
- **pip** package manager
- **(Optional) CUDA** for GPU acceleration

## Step 1: Install Dependencies

### Basic Installation

```bash
# Navigate to project directory
cd /Users/sohamdhirendesai/Desktop/Projects_2/whales

# Install Python dependencies
pip install -r requirements.txt
```

### Alternative: Install with pip

```bash
pip install -e .
```

This installs the package in editable mode and adds the `underwater-flows` command to your PATH.

## Step 2: Verify Installation

```bash
python test_installation.py
```

Expected output:
```
✅ All dependencies available!
✅ All imports successful!
✅ All function tests passed!
✅ CLI test passed!
🎉 Installation successful! You're ready to go.
```

## Step 3: (Optional) Setup RAFT

For production-quality optical flow, install RAFT:

### 3a. Clone RAFT Repository

```bash
cd flows/models
git clone https://github.com/princeton-vl/RAFT.git raft_core
cd ../..
```

### 3b. Download RAFT Weights

Choose one of these pre-trained models:

**RAFT-Things** (trained on FlyingThings3D dataset):
```bash
mkdir -p flows/models
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models/raft-things.pth -O flows/models/raft-things.pth
```

**RAFT-Sintel** (trained on Sintel dataset):
```bash
wget https://www.dropbox.com/s/8gt9pao3yw7fbnd/models/raft-sintel.pth -O flows/models/raft-sintel.pth
```

**RAFT-KITTI** (trained on KITTI dataset):
```bash
wget https://www.dropbox.com/s/o4dfhsowxa9hpxf/models/raft-kitti.pth -O flows/models/raft-kitti.pth
```

Note: Download links may require updating. Check the [official RAFT repo](https://github.com/princeton-vl/RAFT) for latest weights.

## Step 4: Test Run

### With Dummy Backend (no RAFT required)

```bash
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./test_output \
  --backend dummy \
  --resize_long_edge 480 \
  --stride 2
```

### With RAFT Backend

```bash
python -m flows.run \
  --video input_video.mp4 \
  --out_dir ./test_output \
  --backend raft \
  --weights flows/models/raft-things.pth \
  --resize_long_edge 960
```

## Troubleshooting

### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python opencv-contrib-python
```

The contrib version is required for guided filter in transmission estimation.

### Issue: "No module named 'torch'"

**Solution:**
```bash
# For CPU-only
pip install torch torchvision

# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific setup.

### Issue: "RAFT not found"

**Solutions:**
1. Use `--backend dummy` for testing without RAFT
2. Clone RAFT to `flows/models/raft_core` (see Step 3a)
3. Make sure the RAFT directory structure is:
   ```
   flows/models/raft_core/
   ├── core/
   │   ├── raft.py
   │   └── utils/
   └── ...
   ```

### Issue: CUDA Out of Memory

**Solutions:**
1. Reduce resolution: `--resize_long_edge 640` or smaller
2. Use CPU: `--device cpu`
3. Use smaller model: `--backend raft-small`
4. Process fewer frames: `--stride 2` or higher

### Issue: Slow Processing

**Solutions:**
1. Enable GPU: Make sure PyTorch with CUDA is installed
2. Reduce resolution: `--resize_long_edge 640`
3. Skip frames: `--stride 2`
4. Use fast transmission: `--transmission_method fast`
5. Disable scene flow: `--scene_flow_track none`

### Issue: ImportError for ximgproc

The guided filter requires opencv-contrib. Install it:
```bash
pip install opencv-contrib-python
```

If you still have issues, the transmission estimation will fall back to non-guided filtering.

## Platform-Specific Notes

### macOS

- Apple Silicon (M1/M2/M3): PyTorch has native ARM support
- Install with: `pip install torch torchvision`
- GPU acceleration uses Metal Performance Shaders (MPS)

### Linux

- CUDA is well-supported
- Make sure CUDA drivers are installed: `nvidia-smi`
- Install PyTorch with matching CUDA version

### Windows

- Use Anaconda/Miniconda for easier dependency management
- Install Visual C++ Build Tools if needed
- CUDA installation may require additional drivers

## Verifying GPU Support

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Directory Structure After Installation

```
whales/
├── flows/                      # Main package
│   ├── core/                   # Core processing modules
│   ├── models/                 # Model wrappers
│   │   ├── raft_core/         # RAFT repo (after Step 3a)
│   │   ├── raft_wrapper.py
│   │   └── *.pth              # Model weights (after Step 3b)
│   └── utils/                  # Utilities
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── test_installation.py        # Installation test
├── example_usage.py            # Usage examples
├── README.md                   # Full documentation
├── QUICKSTART.md              # Quick start guide
└── INSTALLATION.md            # This file
```

## Next Steps

After successful installation:

1. ✅ Read [QUICKSTART.md](QUICKSTART.md) for basic usage
2. ✅ Try [example_usage.py](example_usage.py) for API examples
3. ✅ Read full [README.md](README.md) for detailed documentation
4. ✅ Test on your underwater videos!

## Getting Help

- Check existing issues in the repository
- Run `python -m flows.run --help` for CLI options
- Read the troubleshooting section above
- Create a new issue with:
  - Python version: `python --version`
  - PyTorch version: `python -c "import torch; print(torch.__version__)"`
  - Error message and full traceback
  - Operating system

## Uninstallation

To remove the package:

```bash
# If installed with pip install -e .
pip uninstall underwater-flows

# Remove dependencies (optional)
pip uninstall -r requirements.txt

# Remove downloaded models (optional)
rm -rf flows/models/raft_core
rm -f flows/models/*.pth
```

---

Happy flow estimation! 🌊

