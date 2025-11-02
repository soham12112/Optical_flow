"""
Analytical speed estimation from optical flow.
Implements variant 2B: No-depth approach with rotation removal and CSV calibration.

This module converts dense optical flow into forward speed (m/s) estimates using
the motion field equations and a calibration CSV with ground-truth speed.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeedEstimate:
    """Container for speed estimation results."""
    speed_ms: float  # Speed in meters/second
    rel_speed: float  # Relative speed (before scaling)
    scale_factor: float  # Scale factor k applied
    omega: np.ndarray  # Estimated angular velocity (3,)
    confidence: float  # Confidence score (0-1)
    residual_rmse: float  # RMSE of flow residuals after rotation removal
    n_pixels_used: int  # Number of pixels used in estimation


class AnalyticalSpeedEstimator:
    """
    Estimates camera forward speed from optical flow using analytical motion field equations.
    
    For underwater scenes without depth, this uses:
    1. Motion field equations to estimate and remove rotation
    2. Radial flow pattern to compute relative forward speed
    3. CSV calibration to convert to metric speed (m/s)
    """
    
    def __init__(
        self,
        intrinsics,
        use_robust_estimation: bool = True,
        min_pixels: int = 100,
        residual_threshold: float = 0.1,
    ):
        """
        Args:
            intrinsics: Camera intrinsics object with f, cx, cy
            use_robust_estimation: Use robust (median-based) estimators
            min_pixels: Minimum number of valid pixels required
            residual_threshold: Threshold for outlier rejection (in normalized flow units)
        """
        self.intrinsics = intrinsics
        self.use_robust = use_robust_estimation
        self.min_pixels = min_pixels
        self.residual_threshold = residual_threshold
        
        # Scale factor (k = Z_0) - will be calibrated from CSV
        self.scale_factor = None
        self.scale_calibrated = False
    
    def estimate_speed(
        self,
        flow: np.ndarray,
        weights: Optional[np.ndarray] = None,
        dt: float = 1.0,
    ) -> SpeedEstimate:
        """
        Estimate forward speed from optical flow.
        
        Args:
            flow: Optical flow (H, W, 2) in pixels/frame
            weights: Optional weight map (H, W) for reliability/attenuation
            dt: Time between frames in seconds (default: 1.0)
        
        Returns:
            SpeedEstimate with speed in m/s (if calibrated) or relative units
        """
        H, W = flow.shape[:2]
        
        # Create normalized camera coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x, y = self._pixel_to_normalized(u, v)
        
        # Convert flow to angular rates (normalized units per frame)
        dx, dy = self._flow_to_angular_rate(flow)
        
        # Convert to per-second rates
        dx = dx / dt
        dy = dy / dt
        
        # Create weight mask
        if weights is None:
            weights = np.ones((H, W), dtype=np.float32)
        
        # Additional weighting: suppress very small flow (likely noise)
        flow_mag = np.linalg.norm(flow, axis=-1)
        flow_weight = np.where(flow_mag > 0.5, 1.0, 0.1)
        weights = weights * flow_weight
        
        # Flatten arrays for processing
        x_flat = x.flatten()
        y_flat = y.flatten()
        dx_flat = dx.flatten()
        dy_flat = dy.flatten()
        w_flat = weights.flatten()
        
        # Filter by weight threshold
        valid_mask = w_flat > 0.1
        if valid_mask.sum() < self.min_pixels:
            logger.warning(f"Insufficient valid pixels: {valid_mask.sum()} < {self.min_pixels}")
            return self._zero_estimate()
        
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        dx_valid = dx_flat[valid_mask]
        dy_valid = dy_flat[valid_mask]
        w_valid = w_flat[valid_mask]
        
        # Step 1: Estimate and remove rotational component
        omega, rot_residuals = self._estimate_rotation(
            x_valid, y_valid, dx_valid, dy_valid, w_valid
        )
        
        # Compute rotational flow
        dx_rot, dy_rot = self._compute_rotational_flow(x_valid, y_valid, omega)
        
        # Remove rotation to get translational flow
        dx_trans = dx_valid - dx_rot
        dy_trans = dy_valid - dy_rot
        
        # Step 2: Estimate relative forward speed from radial pattern
        rel_speed, speed_residuals = self._estimate_relative_speed(
            x_valid, y_valid, dx_trans, dy_trans, w_valid
        )
        
        # Step 3: Convert to metric speed if calibrated
        if self.scale_calibrated:
            speed_ms = self.scale_factor * rel_speed
        else:
            speed_ms = rel_speed
            logger.debug("Scale factor not calibrated - returning relative speed")
        
        # Compute confidence from residuals
        confidence = self._compute_confidence(speed_residuals, w_valid)
        
        # Compute RMSE of residuals
        residual_rmse = np.sqrt(np.average(speed_residuals**2, weights=w_valid))
        
        return SpeedEstimate(
            speed_ms=speed_ms,
            rel_speed=rel_speed,
            scale_factor=self.scale_factor if self.scale_calibrated else 1.0,
            omega=omega,
            confidence=confidence,
            residual_rmse=residual_rmse,
            n_pixels_used=len(x_valid),
        )
    
    def calibrate_scale(
        self,
        frame_times: np.ndarray,
        relative_speeds: np.ndarray,
        csv_times: np.ndarray,
        csv_speeds: np.ndarray,
        use_robust: bool = True,
        time_offset: Optional[float] = None,
    ) -> float:
        """
        Calibrate scale factor k from CSV ground-truth speeds.
        
        This finds the scalar k that best matches relative speeds to CSV speeds:
            k* = argmin || csv_speed - k * rel_speed ||^2
        
        Args:
            frame_times: Times for each frame (N,) - starts at 0
            relative_speeds: Relative speeds from flow (N,)
            csv_times: Times from CSV (M,) - may be absolute time of day
            csv_speeds: Speeds from CSV in m/s (M,)
            use_robust: Use robust estimation (median-based)
            time_offset: Optional time offset to align CSV times with frame times.
                        If None, automatically computed as csv_times[0] - frame_times[0]
        
        Returns:
            Calibrated scale factor k
        """
        from scipy.interpolate import interp1d
        
        # Align CSV times with frame times
        if time_offset is None:
            # Automatic: assume CSV starts when video starts
            time_offset = csv_times[0] - frame_times[0]
            logger.info(f"Auto-detected time offset: {time_offset:.2f} seconds")
            logger.info(f"CSV absolute time {csv_times[0]:.2f}s → video time {frame_times[0]:.2f}s")
        
        # Align CSV times to video timeline
        csv_times_aligned = csv_times - time_offset
        logger.info(f"Aligned CSV time range: {csv_times_aligned[0]:.2f} - {csv_times_aligned[-1]:.2f} seconds")
        
        # Interpolate CSV speeds at frame times
        if csv_times_aligned[0] > frame_times[-1] or csv_times_aligned[-1] < frame_times[0]:
            logger.error(f"CSV times do not overlap with frame times!")
            logger.error(f"CSV range: {csv_times_aligned[0]:.2f} - {csv_times_aligned[-1]:.2f}s")
            logger.error(f"Frame range: {frame_times[0]:.2f} - {frame_times[-1]:.2f}s")
            logger.error(f"Try adjusting the time offset with --time-offset parameter")
            return 1.0
        
        # Create interpolator (linear, extrapolate at edges)
        interpolator = interp1d(
            csv_times_aligned,
            csv_speeds,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        csv_speeds_interp = interpolator(frame_times)
        
        # Find valid overlapping region
        valid_mask = (
            (frame_times >= csv_times_aligned[0]) &
            (frame_times <= csv_times_aligned[-1]) &
            (relative_speeds > 1e-6) &  # Avoid division by zero
            (csv_speeds_interp > 1e-6)
        )
        
        if valid_mask.sum() < 10:
            logger.warning("Insufficient overlap between frame times and CSV times")
            return 1.0
        
        rel_speeds_valid = relative_speeds[valid_mask]
        csv_speeds_valid = csv_speeds_interp[valid_mask]
        
        if use_robust:
            # Robust estimation: use median of ratios
            ratios = csv_speeds_valid / rel_speeds_valid
            # Remove outliers (> 3 MAD from median)
            median_ratio = np.median(ratios)
            mad = np.median(np.abs(ratios - median_ratio))
            inlier_mask = np.abs(ratios - median_ratio) < 3 * mad
            if inlier_mask.sum() > 5:
                k = np.median(ratios[inlier_mask])
            else:
                k = median_ratio
        else:
            # Least squares: k = (csv^T * rel) / (rel^T * rel)
            k = np.sum(csv_speeds_valid * rel_speeds_valid) / np.sum(rel_speeds_valid**2)
        
        logger.info(f"Calibrated scale factor k = {k:.4f} meters")
        logger.info(f"Used {valid_mask.sum()} frames for calibration")
        
        # Compute calibration quality metrics
        predicted_speeds = k * rel_speeds_valid
        rmse = np.sqrt(np.mean((predicted_speeds - csv_speeds_valid)**2))
        mae = np.mean(np.abs(predicted_speeds - csv_speeds_valid))
        r2 = 1 - np.sum((csv_speeds_valid - predicted_speeds)**2) / np.sum((csv_speeds_valid - csv_speeds_valid.mean())**2)
        
        logger.info(f"Calibration quality: RMSE={rmse:.4f} m/s, MAE={mae:.4f} m/s, R²={r2:.4f}")
        
        self.scale_factor = k
        self.scale_calibrated = True
        
        return k
    
    def _pixel_to_normalized(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel coordinates to normalized camera coordinates."""
        K = self.intrinsics.to_matrix()
        f = K[0, 0]  # Assuming fx = fy
        cx = K[0, 2]
        cy = K[1, 2]
        
        x = (u - cx) / f
        y = (v - cy) / f
        
        return x, y
    
    def _flow_to_angular_rate(
        self,
        flow: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert optical flow (pixels/frame) to angular rates (normalized units/frame)."""
        K = self.intrinsics.to_matrix()
        f = K[0, 0]
        
        dx = flow[..., 0] / f
        dy = flow[..., 1] / f
        
        return dx, dy
    
    def _estimate_rotation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate angular velocity omega from flow using rotational part of motion field.
        
        The rotational component satisfies:
            [dx]   [  x*y      -(1+x²)   y  ] [ωx]
            [dy] = [(1+y²)    -x*y     -x  ] [ωy]
                                              [ωz]
        
        Returns:
            omega: Angular velocity (ωx, ωy, ωz) in rad/s
            residuals: Flow residuals after fitting rotation
        """
        N = len(x)
        
        # Build B matrix (Nx2x3) for rotational flow
        B11 = x * y
        B12 = -(1 + x * x)
        B13 = y
        B21 = 1 + y * y
        B22 = -x * y
        B23 = -x
        
        # Stack into design matrix: each pixel contributes 2 equations
        # A is (2N x 3): [B11 B12 B13]
        #                [B21 B22 B23] for each pixel
        A = np.zeros((2 * N, 3), dtype=np.float32)
        A[0::2, 0] = B11
        A[0::2, 1] = B12
        A[0::2, 2] = B13
        A[1::2, 0] = B21
        A[1::2, 1] = B22
        A[1::2, 2] = B23
        
        # Observations
        b = np.zeros(2 * N, dtype=np.float32)
        b[0::2] = dx
        b[1::2] = dy
        
        # Weights (repeat for each equation)
        W = np.zeros(2 * N, dtype=np.float32)
        W[0::2] = weights
        W[1::2] = weights
        
        # Weighted least squares: omega = (A^T W A)^-1 A^T W b
        if self.use_robust:
            # Initial estimate
            omega = self._solve_weighted_ls(A, b, W)
            
            # Robust reweighting (IRLS with Huber)
            for _ in range(3):
                residuals = b - A @ omega
                residual_std = np.sqrt(np.average(residuals**2, weights=W))
                # Huber weights
                threshold = 1.5 * residual_std
                robust_weights = np.where(
                    np.abs(residuals) <= threshold,
                    1.0,
                    threshold / (np.abs(residuals) + 1e-6)
                )
                W_robust = W * robust_weights
                omega = self._solve_weighted_ls(A, b, W_robust)
        else:
            omega = self._solve_weighted_ls(A, b, W)
        
        # Compute final residuals
        flow_residuals = b - A @ omega
        # Aggregate to per-pixel residuals (L2 norm of 2 equations)
        residuals = np.sqrt(flow_residuals[0::2]**2 + flow_residuals[1::2]**2)
        
        return omega, residuals
    
    def _solve_weighted_ls(
        self,
        A: np.ndarray,
        b: np.ndarray,
        W: np.ndarray,
    ) -> np.ndarray:
        """Solve weighted least squares: x = (A^T W A)^-1 A^T W b"""
        # Weight the system
        W_sqrt = np.sqrt(W)[:, None]
        A_weighted = A * W_sqrt
        b_weighted = b * W_sqrt.squeeze()
        
        # Solve using SVD for numerical stability
        x, residuals, rank, s = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
        
        return x
    
    def _compute_rotational_flow(
        self,
        x: np.ndarray,
        y: np.ndarray,
        omega: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rotational flow at points (x, y) given omega."""
        wx, wy, wz = omega
        
        dx_rot = x * y * wx - (1 + x * x) * wy + y * wz
        dy_rot = (1 + y * y) * wx - x * y * wy - x * wz
        
        return dx_rot, dy_rot
    
    def _estimate_relative_speed(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dx_trans: np.ndarray,
        dy_trans: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Estimate relative forward speed from translational flow using radial pattern.
        
        For forward motion V with dominant depth Z₀:
            √(dx² + dy²) ≈ (V / Z₀) * √(x² + y²)
        
        So: V_rel = V / Z₀ = median( |flow| / r )
        
        Returns:
            rel_speed: Relative speed V_rel (units: 1/s)
            residuals: Per-pixel residuals
        """
        # Radial distance from principal point
        r = np.sqrt(x * x + y * y) + 1e-6
        
        # Translational flow magnitude
        flow_mag = np.sqrt(dx_trans * dx_trans + dy_trans * dy_trans)
        
        # Relative speed estimate per pixel
        v_rel_per_pixel = flow_mag / r
        
        if self.use_robust:
            # Robust median estimator
            rel_speed = self._weighted_median(v_rel_per_pixel, weights)
            
            # Compute residuals
            predicted_flow_mag = rel_speed * r
            residuals = np.abs(flow_mag - predicted_flow_mag)
        else:
            # Weighted mean
            rel_speed = np.average(v_rel_per_pixel, weights=weights)
            predicted_flow_mag = rel_speed * r
            residuals = np.abs(flow_mag - predicted_flow_mag)
        
        return rel_speed, residuals
    
    def _weighted_median(
        self,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute weighted median."""
        # Sort by values
        sorted_idx = np.argsort(values)
        values_sorted = values[sorted_idx]
        weights_sorted = weights[sorted_idx]
        
        # Compute cumulative weights
        cum_weights = np.cumsum(weights_sorted)
        total_weight = cum_weights[-1]
        
        # Find median (50th percentile)
        median_idx = np.searchsorted(cum_weights, 0.5 * total_weight)
        median_idx = min(median_idx, len(values_sorted) - 1)
        
        return values_sorted[median_idx]
    
    def _compute_confidence(
        self,
        residuals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute confidence score from residuals (0 = low, 1 = high)."""
        # Use negative exponential of weighted residual
        weighted_residual = np.average(residuals, weights=weights)
        confidence = np.exp(-5.0 * weighted_residual)  # Tuned decay
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _zero_estimate(self) -> SpeedEstimate:
        """Return zero estimate when computation fails."""
        return SpeedEstimate(
            speed_ms=0.0,
            rel_speed=0.0,
            scale_factor=self.scale_factor if self.scale_calibrated else 1.0,
            omega=np.zeros(3),
            confidence=0.0,
            residual_rmse=0.0,
            n_pixels_used=0,
        )


def load_speed_csv(
    csv_path: str,
    time_col: str = "time",
    speed_col: str = "speed",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load time and speed from CSV file.
    
    Args:
        csv_path: Path to CSV file
        time_col: Name of time column
        speed_col: Name of speed column
    
    Returns:
        Tuple of (times, speeds) as numpy arrays
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in CSV. Available: {df.columns.tolist()}")
    if speed_col not in df.columns:
        raise ValueError(f"Speed column '{speed_col}' not found in CSV. Available: {df.columns.tolist()}")
    
    # Parse time column (handle MM:SS.S or HH:MM:SS.S format)
    def parse_time(time_str):
        """Convert time string to seconds."""
        if pd.isna(time_str):
            return np.nan
        
        time_str = str(time_str).strip()
        
        # Try direct numeric conversion first
        try:
            return float(time_str)
        except ValueError:
            pass
        
        # Try MM:SS.S or HH:MM:SS.S format
        try:
            parts = time_str.split(':')
            if len(parts) == 2:  # MM:SS.S
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            elif len(parts) == 3:  # HH:MM:SS.S
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        except (ValueError, AttributeError):
            pass
        
        return np.nan
    
    # Parse times and speeds
    times = df[time_col].apply(parse_time).values
    speeds = pd.to_numeric(df[speed_col], errors='coerce').values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(times) | np.isnan(speeds))
    times = times[valid_mask]
    speeds = speeds[valid_mask]
    
    if len(times) == 0:
        raise ValueError(
            f"No valid numeric data found in CSV columns '{time_col}' and '{speed_col}'.\n"
            f"Time format should be either decimal seconds (e.g., '10.5') or MM:SS.S (e.g., '1:30.5')"
        )
    
    logger.info(f"Loaded {len(times)} speed measurements from {csv_path}")
    logger.info(f"CSV time range (absolute): {times[0]:.2f} - {times[-1]:.2f} seconds")
    logger.info(f"Speed range: {speeds.min():.2f} - {speeds.max():.2f} m/s")
    
    return times, speeds


def smooth_speeds(
    speeds: np.ndarray,
    window_size: int = 5,
    method: str = "savgol",
) -> np.ndarray:
    """
    Smooth speed estimates over time.
    
    Args:
        speeds: Speed estimates (N,)
        window_size: Window size for smoothing
        method: Smoothing method ('savgol', 'gaussian', 'median')
    
    Returns:
        Smoothed speeds
    """
    if method == "savgol":
        from scipy.signal import savgol_filter
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        window_size = min(window_size, len(speeds))
        if window_size < 3:
            return speeds
        return savgol_filter(speeds, window_size, polyorder=2)
    
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 3.0
        return gaussian_filter1d(speeds, sigma)
    
    elif method == "median":
        from scipy.ndimage import median_filter
        return median_filter(speeds, size=window_size)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")

