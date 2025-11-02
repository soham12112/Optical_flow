#!/usr/bin/env python3
"""
Quick verification that speed estimation module is properly installed.
Tests imports and basic functionality without requiring matplotlib.
"""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError as e:
        print(f"  ✗ scipy: {e}")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False
    
    try:
        from flows.core.speed_estimation import AnalyticalSpeedEstimator
        print("  ✓ speed_estimation module")
    except ImportError as e:
        print(f"  ✗ speed_estimation module: {e}")
        return False
    
    try:
        from flows.utils.intrinsics import CameraIntrinsics
        print("  ✓ intrinsics module")
    except ImportError as e:
        print(f"  ✗ intrinsics module: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic speed estimation functionality."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from flows.core.speed_estimation import AnalyticalSpeedEstimator
        from flows.utils.intrinsics import CameraIntrinsics
        
        # Create simple test case
        intrinsics = CameraIntrinsics(
            fx=1000.0,
            fy=1000.0,
            cx=500.0,
            cy=500.0,
        )
        
        estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)
        print("  ✓ Created estimator")
        
        # Create dummy flow
        flow = np.random.randn(100, 100, 2).astype(np.float32) * 0.5
        weights = np.ones((100, 100), dtype=np.float32)
        
        estimate = estimator.estimate_speed(flow, weights, dt=1.0)
        print(f"  ✓ Computed speed estimate: {estimate.rel_speed:.6f}")
        print(f"    - Confidence: {estimate.confidence:.3f}")
        print(f"    - Omega: [{estimate.omega[0]:.3f}, {estimate.omega[1]:.3f}, {estimate.omega[2]:.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_loading():
    """Test CSV loading functionality."""
    print("\nTesting CSV loading...")
    
    try:
        import numpy as np
        import pandas as pd
        from flows.core.speed_estimation import load_speed_csv
        import tempfile
        import os
        
        # Create temporary CSV
        csv_content = """time,speed
0.0,0.5
1.0,1.2
2.0,1.8
3.0,2.1
4.0,1.5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            csv_path = f.name
        
        try:
            times, speeds = load_speed_csv(csv_path)
            print(f"  ✓ Loaded CSV: {len(times)} entries")
            print(f"    - Time range: {times[0]:.1f} - {times[-1]:.1f} s")
            print(f"    - Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
            return True
        finally:
            os.unlink(csv_path)
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration():
    """Test scale calibration."""
    print("\nTesting scale calibration...")
    
    try:
        import numpy as np
        from flows.core.speed_estimation import AnalyticalSpeedEstimator
        from flows.utils.intrinsics import CameraIntrinsics
        
        intrinsics = CameraIntrinsics(fx=1000, fy=1000, cx=500, cy=500)
        estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)
        
        # Simulate data: true_speed = k * rel_speed with k=5.0
        k_true = 5.0
        frame_times = np.arange(0, 10, 0.5)
        true_speeds = 1.0 + 0.5 * np.sin(2 * np.pi * frame_times / 5.0)
        relative_speeds = true_speeds / k_true
        
        # Calibrate
        k_estimated = estimator.calibrate_scale(
            frame_times=frame_times,
            relative_speeds=relative_speeds,
            csv_times=frame_times[::2],
            csv_speeds=true_speeds[::2],
            use_robust=True,
        )
        
        error = abs(k_estimated - k_true)
        error_pct = error / k_true * 100
        
        print(f"  ✓ Calibration successful")
        print(f"    - True k:      {k_true:.4f}")
        print(f"    - Estimated k: {k_estimated:.4f}")
        print(f"    - Error:       {error:.4f} ({error_pct:.2f}%)")
        
        if error_pct < 5.0:
            print(f"    ✓ Error < 5% - Excellent!")
            return True
        elif error_pct < 10.0:
            print(f"    ✓ Error < 10% - Good")
            return True
        else:
            print(f"    ⚠ Error > 10% - May need investigation")
            return False
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("SPEED ESTIMATION MODULE - INSTALLATION VERIFICATION")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports worked
        results.append(("Basic Functionality", test_basic_functionality()))
        results.append(("CSV Loading", test_csv_loading()))
        results.append(("Calibration", test_calibration()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Installation verified! Speed estimation is ready to use.")
        print("\nNext steps:")
        print("  1. Install matplotlib for visualizations:")
        print("     pip install matplotlib")
        print("  2. Run full test suite:")
        print("     python test_speed_estimation.py")
        print("  3. Process your video and estimate speed:")
        print("     python estimate_speed.py outputs --csv calibration.csv")
        return 0
    else:
        print(f"\n⚠️  {total - passed} check(s) failed.")
        print("\nTo fix:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

