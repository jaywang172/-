"""
Simple Drift Detectors - Standalone Implementation
No external dependencies required (except numpy)
"""

import numpy as np
from typing import Dict, Optional, List
from collections import deque


class SimpleADWIN:
    """
    Simplified ADWIN (Adaptive Windowing) implementation
    Detects changes in data distribution using statistical tests
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 50):
        """
        Initialize Simple ADWIN detector

        Args:
            delta: Confidence parameter (smaller = more sensitive)
            max_buckets: Maximum number of buckets to maintain
        """
        self.delta = delta
        self.max_buckets = max_buckets
        self.window = deque(maxlen=1000)  # Sliding window
        self.drift_detected_flag = False
        self.total_count = 0

    def update(self, value: float) -> bool:
        """
        Update detector with new value

        Args:
            value: New observation

        Returns:
            bool: True if drift detected
        """
        self.window.append(value)
        self.total_count += 1
        self.drift_detected_flag = False

        # Need at least 10 samples
        if len(self.window) < 10:
            return False

        # Check for drift by comparing recent vs older data
        if len(self.window) >= 20:
            recent_window = list(self.window)[-10:]
            older_window = list(self.window)[-20:-10]

            # Use mean and variance difference
            recent_mean = np.mean(recent_window)
            older_mean = np.mean(older_window)
            recent_var = np.var(recent_window)
            older_var = np.var(older_window)

            # Compute difference threshold based on delta
            threshold = self.delta * (recent_var + older_var + 0.001)

            mean_diff = abs(recent_mean - older_mean)

            if mean_diff > threshold:
                self.drift_detected_flag = True
                # Keep recent data, clear older
                self.window = deque(recent_window, maxlen=1000)
                return True

        return False

    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected"""
        return self.drift_detected_flag


class SimpleDDM:
    """
    Simplified DDM (Drift Detection Method) implementation
    Monitors error rate and its standard deviation
    """

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        """
        Initialize Simple DDM detector

        Args:
            warning_level: Warning level threshold (standard deviations)
            drift_level: Drift level threshold (standard deviations)
        """
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset_statistics()

    def reset_statistics(self):
        """Reset internal statistics"""
        self.error_count = 0
        self.total_count = 0
        self.min_error_rate = float('inf')
        self.min_std = float('inf')
        self.drift_detected_flag = False
        self.warning_detected_flag = False

    def update(self, error: int) -> bool:
        """
        Update detector with binary error value

        Args:
            error: 1 if error, 0 if correct

        Returns:
            bool: True if drift detected
        """
        self.error_count += error
        self.total_count += 1
        self.drift_detected_flag = False
        self.warning_detected_flag = False

        if self.total_count < 30:
            return False

        # Calculate current error rate and std
        error_rate = self.error_count / self.total_count
        std = np.sqrt(error_rate * (1 - error_rate) / self.total_count)

        # Update minimum values
        if error_rate + std < self.min_error_rate + self.min_std:
            self.min_error_rate = error_rate
            self.min_std = std

        # Check for warning
        if error_rate + std > self.min_error_rate + self.warning_level * self.min_std:
            self.warning_detected_flag = True

        # Check for drift
        if error_rate + std > self.min_error_rate + self.drift_level * self.min_std:
            self.drift_detected_flag = True
            self.reset_statistics()
            return True

        return False

    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected"""
        return self.drift_detected_flag

    @property
    def warning_detected(self) -> bool:
        """Check if warning was detected"""
        return self.warning_detected_flag


class SimpleKSWIN:
    """
    Simplified KSWIN (Kolmogorov-Smirnov Windowing) implementation
    Uses statistical test to compare windows
    """

    def __init__(self, alpha: float = 0.005, window_size: int = 100, stat_size: int = 30):
        """
        Initialize Simple KSWIN detector

        Args:
            alpha: Significance level
            window_size: Size of reference window
            stat_size: Size of recent window for comparison
        """
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.reference_window = deque(maxlen=window_size)
        self.recent_window = deque(maxlen=stat_size)
        self.drift_detected_flag = False

    def update(self, value: float) -> bool:
        """
        Update detector with new value

        Args:
            value: New observation

        Returns:
            bool: True if drift detected
        """
        self.drift_detected_flag = False

        # Add to both windows
        if len(self.reference_window) < self.window_size:
            self.reference_window.append(value)
            return False

        self.recent_window.append(value)

        # Perform test when recent window is full
        if len(self.recent_window) == self.stat_size:
            # Perform simplified KS test
            drift = self._ks_test()
            if drift:
                self.drift_detected_flag = True
                # Replace reference with recent
                self.reference_window = deque(list(self.recent_window), maxlen=self.window_size)
                self.recent_window.clear()
                return True

        return False

    def _ks_test(self) -> bool:
        """
        Simplified Kolmogorov-Smirnov test

        Returns:
            bool: True if distributions are significantly different
        """
        ref_data = np.array(list(self.reference_window))
        recent_data = np.array(list(self.recent_window))

        # Calculate empirical CDFs and max difference
        ref_sorted = np.sort(ref_data)
        recent_sorted = np.sort(recent_data)

        # Use percentiles for simplified comparison
        ref_percentiles = np.percentile(ref_sorted, [25, 50, 75])
        recent_percentiles = np.percentile(recent_sorted, [25, 50, 75])

        # Check if percentiles differ significantly
        max_diff = np.max(np.abs(ref_percentiles - recent_percentiles))

        # Threshold based on alpha
        threshold = 1.36 * self.alpha  # Simplified threshold

        return max_diff > threshold

    @property
    def drift_detected(self) -> bool:
        """Check if drift was detected"""
        return self.drift_detected_flag


# Re-export with original names for compatibility
ADWIN = SimpleADWIN
DDM = SimpleDDM
KSWIN = SimpleKSWIN
