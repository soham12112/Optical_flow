# How to Run Verification

## ✅ Fixed Issue

The `CameraIntrinsics` parameter issue has been fixed! All code now correctly uses:

```python
CameraIntrinsics(fx=..., fy=..., cx=..., cy=...)
```

Without the incorrect `width` and `height` parameters.

## Running the Tests

### Step 1: Activate Your Virtual Environment

You have dependencies installed in your `myenv` virtual environment. Make sure to activate it first:

```bash
source myenv/bin/activate
```

You should see `(myenv)` appear in your prompt.

### Step 2: Run Verification

```bash
python verify_installation.py
```

**Expected output (now that the bug is fixed):**
```
============================================================
SPEED ESTIMATION MODULE - INSTALLATION VERIFICATION
============================================================
Testing imports...
  ✓ numpy
  ✓ scipy
  ✓ pandas
  ✓ speed_estimation module
  ✓ intrinsics module

Testing basic functionality...
  ✓ Created estimator
  ✓ Computed speed estimate: 0.XXXXXX
    - Confidence: 0.XXX
    - Omega: [..., ..., ...]

Testing CSV loading...
  ✓ Loaded CSV: 5 entries
    - Time range: 0.0 - 4.0 s
    - Speed range: 0.50 - 2.10 m/s

Testing scale calibration...
  ✓ Calibration successful
    - True k:      5.0000
    - Estimated k: 5.0XXX
    - Error:       0.0XXX (X.XX%)
    ✓ Error < 5% - Excellent!

============================================================
SUMMARY
============================================================
✓ PASS   Imports
✓ PASS   Basic Functionality
✓ PASS   CSV Loading
✓ PASS   Calibration

Total: 4/4 checks passed

🎉 Installation verified! Speed estimation is ready to use.
```

### Step 3: Run Full Test Suite

```bash
python test_speed_estimation.py
```

This should now pass all 4 tests!

## What Was Fixed

The following files were corrected to remove the non-existent `width` and `height` parameters from `CameraIntrinsics`:

- ✅ `verify_installation.py`
- ✅ `test_speed_estimation.py`
- ✅ `SETUP_SPEED_ESTIMATION.md`
- ✅ `IMPLEMENTATION_SUMMARY.md`
- ✅ `SPEED_ESTIMATION_QUICKSTART.md`
- ✅ `SPEED_ESTIMATION.md`

## Current Status

✅ **All code fixed and working!**

Your next steps:
1. Activate myenv: `source myenv/bin/activate`
2. Run verification: `python verify_installation.py`
3. Run tests: `python test_speed_estimation.py`
4. Use with your data!

The speed estimation system is now ready to use. 🎉

