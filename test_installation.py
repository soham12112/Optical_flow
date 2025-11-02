#!/usr/bin/env python3
"""
Quick installation test script.
Tests that all modules import correctly.
"""

import sys

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        print("  ✓ Testing flows package...")
        import flows
        print(f"    Version: {flows.__version__}")
        
        print("  ✓ Testing core modules...")
        from flows.core import frame_extraction
        from flows.core import underwater_preproc
        from flows.core import transmission
        from flows.core import optical_flow
        from flows.core import scene_flow
        from flows.core import processor
        
        print("  ✓ Testing utils modules...")
        from flows.utils import intrinsics
        from flows.utils import visualization
        
        print("  ✓ Testing models...")
        from flows.models import raft_wrapper
        
        print("  ✓ Testing API...")
        from flows import process_video, load_intrinsics
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_basic_functions():
    """Test basic functionality."""
    print("\nTesting basic functions...")
    
    try:
        import numpy as np
        from flows.utils.intrinsics import CameraIntrinsics
        from flows.utils.visualization import flow_to_color
        from flows.core.transmission import estimate_transmission_fast
        from flows.core.underwater_preproc import apply_gamma_correction
        
        # Test intrinsics
        print("  ✓ Testing intrinsics...")
        intrinsics = CameraIntrinsics(fx=500, fy=500, cx=320, cy=240)
        K = intrinsics.to_matrix()
        assert K.shape == (3, 3)
        
        # Test visualization
        print("  ✓ Testing flow visualization...")
        flow = np.random.randn(100, 100, 2).astype(np.float32)
        flow_vis = flow_to_color(flow)
        assert flow_vis.shape == (100, 100, 3)
        
        # Test transmission estimation
        print("  ✓ Testing transmission estimation...")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        transmission = estimate_transmission_fast(image)
        assert transmission.shape == (100, 100)
        assert transmission.min() >= 0 and transmission.max() <= 1
        
        # Test gamma correction
        print("  ✓ Testing gamma correction...")
        corrected = apply_gamma_correction(image, gamma=0.8)
        assert corrected.shape == image.shape
        
        print("\n✅ All function tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """Check that key dependencies are available."""
    print("\nChecking dependencies...")
    
    deps = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "torch": "torch",
        "tqdm": "tqdm",
        "PIL": "pillow",
    }
    
    missing = []
    for module, package in deps.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True


def test_cli():
    """Test CLI interface."""
    print("\nTesting CLI...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "flows.run", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0 and "--video" in result.stdout:
            print("  ✓ CLI help works")
            print("\n✅ CLI test passed!")
            return True
        else:
            print(f"  ❌ CLI returned error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n❌ CLI test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Underwater Flows Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("Functions", test_basic_functions()))
    results.append(("CLI", test_cli()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 Installation successful! You're ready to go.")
        print("\nNext steps:")
        print("  1. Test with dummy backend:")
        print("     python -m flows.run --video input_video.mp4 --backend dummy")
        print("  2. (Optional) Setup RAFT for real optical flow")
        print("  3. Check out example_usage.py for API examples")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

