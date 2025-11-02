#!/usr/bin/env python3
"""
Test speed estimation with synthetic optical flow data.

This script creates synthetic flow fields with known motion parameters
and verifies that the speed estimator can recover them correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from flows.core.speed_estimation import AnalyticalSpeedEstimator
from flows.utils.intrinsics import CameraIntrinsics


def generate_synthetic_flow(
    width: int,
    height: int,
    focal_length: float,
    velocity: np.ndarray,  # [Tx, Ty, Tz] in m/s
    omega: np.ndarray,  # [ωx, ωy, ωz] in rad/s
    depth: float = 5.0,  # Average depth in meters
) -> np.ndarray:
    """
    Generate synthetic optical flow from motion parameters.
    
    Uses motion field equations:
        ẋ = (-Tx + x*Tz)/Z + x*y*ωx - (1+x²)*ωy + y*ωz
        ẏ = (-Ty + y*Tz)/Z + (1+y²)*ωx - x*y*ωy - x*ωz
    """
    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to normalized coordinates
    cx, cy = width / 2, height / 2
    x = (u - cx) / focal_length
    y = (v - cy) / focal_length
    
    # Unpack motion parameters
    Tx, Ty, Tz = velocity
    wx, wy, wz = omega
    
    # Translational flow (normalized)
    dx_trans = (-Tx + x * Tz) / depth
    dy_trans = (-Ty + y * Tz) / depth
    
    # Rotational flow (normalized)
    dx_rot = x * y * wx - (1 + x * x) * wy + y * wz
    dy_rot = (1 + y * y) * wx - x * y * wy - x * wz
    
    # Total flow (normalized)
    dx = dx_trans + dx_rot
    dy = dy_trans + dy_rot
    
    # Convert back to pixels
    flow_u = dx * focal_length
    flow_v = dy * focal_length
    
    flow = np.stack([flow_u, flow_v], axis=-1)
    
    return flow


def test_forward_motion():
    """Test 1: Pure forward motion (no rotation)"""
    print("\n" + "="*60)
    print("TEST 1: Pure Forward Motion")
    print("="*60)
    
    # Setup
    width, height = 1920, 1080
    focal_length = 1200.0
    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width/2,
        cy=height/2,
    )
    
    # Ground truth motion: forward at 2 m/s
    velocity_true = np.array([0.0, 0.0, 2.0])  # Forward along Z axis
    omega_true = np.array([0.0, 0.0, 0.0])  # No rotation
    depth = 5.0  # 5 meters depth
    
    # Generate synthetic flow
    flow = generate_synthetic_flow(
        width, height, focal_length,
        velocity_true, omega_true, depth
    )
    
    # Create estimator
    estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)
    
    # Estimate speed (without calibration - relative units)
    weights = np.ones((height, width), dtype=np.float32)
    estimate = estimator.estimate_speed(flow, weights, dt=1.0)
    
    # Expected relative speed: V_rel = Tz / Z = 2.0 / 5.0 = 0.4
    expected_rel_speed = velocity_true[2] / depth
    
    print(f"Ground truth:")
    print(f"  Velocity:      [{velocity_true[0]:.2f}, {velocity_true[1]:.2f}, {velocity_true[2]:.2f}] m/s")
    print(f"  Omega:         [{omega_true[0]:.3f}, {omega_true[1]:.3f}, {omega_true[2]:.3f}] rad/s")
    print(f"  Depth:         {depth:.1f} m")
    print(f"  Expected V_rel: {expected_rel_speed:.4f}")
    print(f"\nEstimated:")
    print(f"  Rel speed:     {estimate.rel_speed:.4f}")
    print(f"  Omega:         [{estimate.omega[0]:.3f}, {estimate.omega[1]:.3f}, {estimate.omega[2]:.3f}] rad/s")
    print(f"  Confidence:    {estimate.confidence:.3f}")
    print(f"  Residual RMSE: {estimate.residual_rmse:.6f}")
    print(f"\nError:")
    print(f"  Rel speed:     {abs(estimate.rel_speed - expected_rel_speed):.6f} ({abs(estimate.rel_speed - expected_rel_speed)/expected_rel_speed*100:.2f}%)")
    
    success = abs(estimate.rel_speed - expected_rel_speed) / expected_rel_speed < 0.05
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Relative speed within 5%")
    
    return success


def test_forward_with_rotation():
    """Test 2: Forward motion with rotation"""
    print("\n" + "="*60)
    print("TEST 2: Forward Motion + Rotation")
    print("="*60)
    
    # Setup
    width, height = 1920, 1080
    focal_length = 1200.0
    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width/2,
        cy=height/2,
    )
    
    # Ground truth motion: forward at 2 m/s + yaw rotation
    velocity_true = np.array([0.0, 0.0, 2.0])
    omega_true = np.array([0.0, 0.2, 0.0])  # Yaw at 0.2 rad/s
    depth = 5.0
    
    # Generate synthetic flow
    flow = generate_synthetic_flow(
        width, height, focal_length,
        velocity_true, omega_true, depth
    )
    
    # Create estimator
    estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)
    
    # Estimate speed
    weights = np.ones((height, width), dtype=np.float32)
    estimate = estimator.estimate_speed(flow, weights, dt=1.0)
    
    expected_rel_speed = velocity_true[2] / depth
    
    print(f"Ground truth:")
    print(f"  Velocity:      [{velocity_true[0]:.2f}, {velocity_true[1]:.2f}, {velocity_true[2]:.2f}] m/s")
    print(f"  Omega:         [{omega_true[0]:.3f}, {omega_true[1]:.3f}, {omega_true[2]:.3f}] rad/s")
    print(f"  Depth:         {depth:.1f} m")
    print(f"  Expected V_rel: {expected_rel_speed:.4f}")
    print(f"\nEstimated:")
    print(f"  Rel speed:     {estimate.rel_speed:.4f}")
    print(f"  Omega:         [{estimate.omega[0]:.3f}, {estimate.omega[1]:.3f}, {estimate.omega[2]:.3f}] rad/s")
    print(f"  Confidence:    {estimate.confidence:.3f}")
    print(f"  Residual RMSE: {estimate.residual_rmse:.6f}")
    print(f"\nError:")
    print(f"  Rel speed:     {abs(estimate.rel_speed - expected_rel_speed):.6f} ({abs(estimate.rel_speed - expected_rel_speed)/expected_rel_speed*100:.2f}%)")
    print(f"  Omega Y:       {abs(estimate.omega[1] - omega_true[1]):.6f} ({abs(estimate.omega[1] - omega_true[1])/abs(omega_true[1])*100:.2f}%)")
    
    success_speed = abs(estimate.rel_speed - expected_rel_speed) / expected_rel_speed < 0.10
    success_omega = abs(estimate.omega[1] - omega_true[1]) / abs(omega_true[1]) < 0.10
    success = success_speed and success_omega
    
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Speed and rotation within 10%")
    
    return success


def test_calibration():
    """Test 3: Scale calibration from CSV"""
    print("\n" + "="*60)
    print("TEST 3: Scale Factor Calibration")
    print("="*60)
    
    # Setup
    width, height = 1920, 1080
    focal_length = 1200.0
    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width/2,
        cy=height/2,
    )
    
    # Simulate a time series with varying speed
    depth = 5.0  # Constant depth
    frame_times = np.arange(0, 10, 0.1)  # 100 frames over 10 seconds
    
    # Ground truth speeds (varying sinusoidally)
    true_speeds = 1.5 + 0.5 * np.sin(2 * np.pi * frame_times / 5.0)  # 1-2 m/s
    
    # Generate relative speeds (as if estimated from flow)
    relative_speeds = true_speeds / depth  # V_rel = V / Z
    
    # Add some noise
    np.random.seed(42)
    relative_speeds += np.random.normal(0, 0.01, len(relative_speeds))
    
    # Create "CSV" data (subsample for calibration)
    csv_times = frame_times[::10]  # Every 10th frame
    csv_speeds = true_speeds[::10]
    
    # Calibrate
    estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics)
    k = estimator.calibrate_scale(
        frame_times=frame_times,
        relative_speeds=relative_speeds,
        csv_times=csv_times,
        csv_speeds=csv_speeds,
        use_robust=True,
    )
    
    print(f"Ground truth depth (Z):  {depth:.2f} m")
    print(f"Calibrated scale (k):    {k:.2f} m")
    print(f"Error:                   {abs(k - depth):.4f} m ({abs(k - depth)/depth*100:.2f}%)")
    
    # Verify metric speeds
    metric_speeds = k * relative_speeds
    rmse = np.sqrt(np.mean((metric_speeds - true_speeds)**2))
    mae = np.mean(np.abs(metric_speeds - true_speeds))
    
    print(f"\nSpeed estimation quality:")
    print(f"  RMSE:                  {rmse:.4f} m/s")
    print(f"  MAE:                   {mae:.4f} m/s")
    
    success = abs(k - depth) / depth < 0.05 and rmse < 0.1
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Calibration within 5%, RMSE < 0.1 m/s")
    
    return success


def test_with_noise():
    """Test 4: Robustness to noise"""
    print("\n" + "="*60)
    print("TEST 4: Robustness to Noisy Flow")
    print("="*60)
    
    # Setup
    width, height = 1920, 1080
    focal_length = 1200.0
    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width/2,
        cy=height/2,
    )
    
    # Ground truth motion
    velocity_true = np.array([0.0, 0.0, 2.0])
    omega_true = np.array([0.0, 0.1, 0.0])
    depth = 5.0
    
    # Generate clean flow
    flow_clean = generate_synthetic_flow(
        width, height, focal_length,
        velocity_true, omega_true, depth
    )
    
    # Add noise (10% of mean flow magnitude)
    np.random.seed(42)
    flow_mag = np.linalg.norm(flow_clean, axis=-1).mean()
    noise_std = 0.1 * flow_mag
    noise = np.random.normal(0, noise_std, flow_clean.shape)
    flow_noisy = flow_clean + noise
    
    # Estimate with clean flow
    estimator = AnalyticalSpeedEstimator(intrinsics=intrinsics, use_robust_estimation=True)
    weights = np.ones((height, width), dtype=np.float32)
    
    estimate_clean = estimator.estimate_speed(flow_clean, weights, dt=1.0)
    estimate_noisy = estimator.estimate_speed(flow_noisy, weights, dt=1.0)
    
    expected_rel_speed = velocity_true[2] / depth
    
    print(f"Expected rel speed:      {expected_rel_speed:.4f}")
    print(f"Clean flow estimate:     {estimate_clean.rel_speed:.4f} (error: {abs(estimate_clean.rel_speed - expected_rel_speed)/expected_rel_speed*100:.2f}%)")
    print(f"Noisy flow estimate:     {estimate_noisy.rel_speed:.4f} (error: {abs(estimate_noisy.rel_speed - expected_rel_speed)/expected_rel_speed*100:.2f}%)")
    print(f"Noise level:             {noise_std:.2f} pixels ({noise_std/flow_mag*100:.1f}% of mean flow)")
    print(f"\nConfidence scores:")
    print(f"  Clean:                 {estimate_clean.confidence:.3f}")
    print(f"  Noisy:                 {estimate_noisy.confidence:.3f}")
    
    # Success if noisy estimate is within 15% (more tolerance for noise)
    success = abs(estimate_noisy.rel_speed - expected_rel_speed) / expected_rel_speed < 0.15
    print(f"\n{'✓ PASS' if success else '✗ FAIL'}: Noisy estimate within 15%")
    
    return success


def visualize_flow_decomposition():
    """Create visualization of flow decomposition (total, rotation, translation)"""
    print("\n" + "="*60)
    print("VISUALIZATION: Flow Decomposition")
    print("="*60)
    
    # Setup (smaller for visualization)
    width, height = 640, 360
    focal_length = 400.0
    intrinsics = CameraIntrinsics(
        fx=focal_length,
        fy=focal_length,
        cx=width/2,
        cy=height/2,
    )
    
    # Motion: forward + yaw
    velocity = np.array([0.0, 0.0, 2.0])
    omega = np.array([0.0, 0.3, 0.0])
    depth = 5.0
    
    # Generate flows
    flow_total = generate_synthetic_flow(width, height, focal_length, velocity, omega, depth)
    flow_trans = generate_synthetic_flow(width, height, focal_length, velocity, np.zeros(3), depth)
    flow_rot = flow_total - flow_trans
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Subsample for quiver plot
    step = 20
    u, v = np.meshgrid(np.arange(0, width, step), np.arange(0, height, step))
    
    def plot_flow(ax, flow, title):
        flow_sub = flow[::step, ::step]
        ax.quiver(u, v, flow_sub[..., 0], flow_sub[..., 1], 
                  angles='xy', scale_units='xy', scale=0.5, color='red')
        ax.set_xlim([0, width])
        ax.set_ylim([height, 0])
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add magnitude colormap
        mag = np.linalg.norm(flow, axis=-1)
        im = ax.imshow(mag, cmap='viridis', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Magnitude (pixels)')
    
    plot_flow(axes[0, 0], flow_total, 'Total Flow')
    plot_flow(axes[0, 1], flow_rot, 'Rotational Component')
    plot_flow(axes[0, 2], flow_trans, 'Translational Component')
    
    # Bottom row: magnitude profiles
    center_y = height // 2
    x_coords = np.arange(width)
    
    mag_total = np.linalg.norm(flow_total[center_y, :], axis=-1)
    mag_rot = np.linalg.norm(flow_rot[center_y, :], axis=-1)
    mag_trans = np.linalg.norm(flow_trans[center_y, :], axis=-1)
    
    axes[1, 0].plot(x_coords, mag_total, 'b-', linewidth=2, label='Total')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Flow magnitude (pixels)')
    axes[1, 0].set_title('Horizontal Profile (center row)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(x_coords, mag_rot, 'r-', linewidth=2, label='Rotation')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_title('Rotational Component Profile')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    axes[1, 2].plot(x_coords, mag_trans, 'g-', linewidth=2, label='Translation')
    axes[1, 2].set_xlabel('X (pixels)')
    axes[1, 2].set_title('Translational Component Profile')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save
    output_path = 'speed_estimation_test_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.show()


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" ANALYTICAL SPEED ESTIMATION - TEST SUITE")
    print("="*70)
    print("\nThis test suite validates the speed estimation module using")
    print("synthetic optical flow with known ground truth motion parameters.")
    print("="*70)
    
    # Run tests
    results = []
    
    try:
        results.append(("Forward Motion", test_forward_motion()))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("Forward Motion", False))
    
    try:
        results.append(("Forward + Rotation", test_forward_with_rotation()))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("Forward + Rotation", False))
    
    try:
        results.append(("Calibration", test_calibration()))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("Calibration", False))
    
    try:
        results.append(("Noise Robustness", test_with_noise()))
    except Exception as e:
        print(f"✗ ERROR: {e}")
        results.append(("Noise Robustness", False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Speed estimation is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")
    
    # Generate visualization
    print("\n" + "="*70)
    try:
        visualize_flow_decomposition()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("="*70)


if __name__ == "__main__":
    main()

