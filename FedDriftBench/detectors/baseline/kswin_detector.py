"""
KSWIN (Kolmogorov-Smirnov Windowing) Drift Detector
Uses statistical test to compare recent and reference windows
"""

from typing import Dict, Optional

# Try to import from river, fall back to simple implementation
try:
    from river.drift import KSWIN
except (ImportError, AttributeError):
    from .simple_detectors import KSWIN


class KSWINDetector:
    """
    KSWIN drift detector for federated learning
    """

    def __init__(self, alpha: float = 0.005, window_size: int = 100,
                 stat_size: int = 30):
        """
        Initialize KSWIN detector

        Args:
            alpha: Significance level for KS test
            window_size: Size of the sliding window
            stat_size: Size of the statistic window
        """
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.kswin = KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)
        self.drift_detected = False
        self.drift_timestamps = []
        self.update_count = 0

    def update(self, value: float, timestamp: Optional[int] = None) -> bool:
        """
        Update detector with new value

        Args:
            value: Metric value
            timestamp: Optional timestamp

        Returns:
            bool: True if drift detected
        """
        self.update_count += 1
        if timestamp is None:
            timestamp = self.update_count

        # Update KSWIN
        self.kswin.update(value)

        # Check drift
        if self.kswin.drift_detected:
            self.drift_detected = True
            self.drift_timestamps.append(timestamp)
            return True

        return False

    def reset(self):
        """Reset the detector"""
        self.kswin = KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size
        )
        self.drift_detected = False
        self.update_count = 0

    def get_drift_info(self) -> Dict:
        """Get information about detected drifts"""
        return {
            'detector': 'KSWIN',
            'drift_count': len(self.drift_timestamps),
            'drift_timestamps': self.drift_timestamps,
            'total_updates': self.update_count,
            'parameters': {
                'alpha': self.alpha,
                'window_size': self.window_size,
                'stat_size': self.stat_size
            }
        }


class FederatedKSWIN:
    """KSWIN detector adapted for federated learning"""

    def __init__(self, num_clients: int,
                 alpha: float = 0.005,
                 window_size: int = 100,
                 stat_size: int = 30,
                 global_detector: bool = True):
        """Initialize Federated KSWIN"""
        self.num_clients = num_clients
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.global_detector_enabled = global_detector

        # Client-level detectors
        self.client_detectors = {
            i: KSWINDetector(
                alpha=alpha,
                window_size=window_size,
                stat_size=stat_size
            )
            for i in range(num_clients)
        }

        # Global detector
        if global_detector:
            self.global_detector = KSWINDetector(
                alpha=alpha,
                window_size=window_size,
                stat_size=stat_size
            )
        else:
            self.global_detector = None

    def update_client(self, client_id: int, value: float,
                     timestamp: Optional[int] = None) -> bool:
        """Update detector for a specific client"""
        if client_id not in self.client_detectors:
            return False

        return self.client_detectors[client_id].update(value, timestamp)

    def update_global(self, value: float,
                     timestamp: Optional[int] = None) -> bool:
        """Update global detector"""
        if self.global_detector is None:
            return False

        return self.global_detector.update(value, timestamp)

    def get_drifting_clients(self) -> list:
        """Get list of clients that detected drift"""
        return [
            client_id
            for client_id, detector in self.client_detectors.items()
            if detector.drift_detected
        ]

    def get_all_drift_info(self) -> Dict:
        """Get drift information for all clients"""
        info = {
            'detector': 'Federated-KSWIN',
            'num_clients': self.num_clients,
            'clients': {
                client_id: detector.get_drift_info()
                for client_id, detector in self.client_detectors.items()
            }
        }

        if self.global_detector is not None:
            info['global'] = self.global_detector.get_drift_info()

        return info

    def reset_all(self):
        """Reset all detectors"""
        for detector in self.client_detectors.values():
            detector.reset()
        if self.global_detector is not None:
            self.global_detector.reset()
