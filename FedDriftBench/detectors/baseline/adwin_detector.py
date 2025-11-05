"""
ADWIN (Adaptive Windowing) Drift Detector
Uses adaptive sliding window to detect changes in data distribution
"""

from typing import Dict, Optional

# Try to import from river, fall back to simple implementation
try:
    from river.drift import ADWIN
except (ImportError, AttributeError):
    from .simple_detectors import ADWIN


class ADWINDetector:
    """
    ADWIN drift detector for federated learning
    Monitors performance metrics to detect drift
    """

    def __init__(self, delta: float = 0.002):
        """
        Initialize ADWIN detector

        Args:
            delta: Confidence parameter (smaller = more sensitive)
        """
        self.delta = delta
        self.adwin = ADWIN(delta=delta)
        self.drift_detected = False
        self.drift_timestamps = []
        self.update_count = 0

    def update(self, value: float, timestamp: Optional[int] = None) -> bool:
        """
        Update detector with new value (e.g., error rate or accuracy)

        Args:
            value: Metric value (typically error rate: 1 - accuracy)
            timestamp: Optional timestamp for tracking

        Returns:
            bool: True if drift detected
        """
        self.update_count += 1
        if timestamp is None:
            timestamp = self.update_count

        # Update ADWIN with the value
        self.adwin.update(value)

        # Check if drift was detected
        if self.adwin.drift_detected:
            self.drift_detected = True
            self.drift_timestamps.append(timestamp)
            return True

        return False

    def reset(self):
        """Reset the detector"""
        self.adwin = ADWIN(delta=self.delta)
        self.drift_detected = False
        self.update_count = 0

    def get_drift_info(self) -> Dict:
        """Get information about detected drifts"""
        return {
            'detector': 'ADWIN',
            'drift_count': len(self.drift_timestamps),
            'drift_timestamps': self.drift_timestamps,
            'total_updates': self.update_count,
            'parameters': {
                'delta': self.delta
            }
        }


class FederatedADWIN:
    """
    ADWIN detector adapted for federated learning
    Maintains separate detectors for each client
    """

    def __init__(self, num_clients: int, delta: float = 0.002,
                 global_detector: bool = True):
        """
        Initialize Federated ADWIN

        Args:
            num_clients: Number of clients
            delta: Confidence parameter
            global_detector: Whether to use a global detector for aggregated metrics
        """
        self.num_clients = num_clients
        self.delta = delta
        self.global_detector_enabled = global_detector

        # Client-level detectors
        self.client_detectors = {
            i: ADWINDetector(delta=delta)
            for i in range(num_clients)
        }

        # Global detector
        if global_detector:
            self.global_detector = ADWINDetector(delta=delta)
        else:
            self.global_detector = None

    def update_client(self, client_id: int, error_rate: float,
                     timestamp: Optional[int] = None) -> bool:
        """
        Update detector for a specific client

        Args:
            client_id: ID of the client
            error_rate: Error rate (1 - accuracy)
            timestamp: Optional timestamp

        Returns:
            bool: True if drift detected for this client
        """
        if client_id not in self.client_detectors:
            return False

        return self.client_detectors[client_id].update(error_rate, timestamp)

    def update_global(self, global_error_rate: float,
                     timestamp: Optional[int] = None) -> bool:
        """
        Update global detector with aggregated metric

        Args:
            global_error_rate: Global error rate
            timestamp: Optional timestamp

        Returns:
            bool: True if drift detected globally
        """
        if self.global_detector is None:
            return False

        return self.global_detector.update(global_error_rate, timestamp)

    def get_drifting_clients(self) -> list:
        """Get list of clients that detected drift"""
        return [
            client_id
            for client_id, detector in self.client_detectors.items()
            if detector.drift_detected
        ]

    def get_all_drift_info(self) -> Dict:
        """Get drift information for all clients and global detector"""
        info = {
            'detector': 'Federated-ADWIN',
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
