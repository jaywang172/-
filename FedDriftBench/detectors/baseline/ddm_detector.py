"""
DDM (Drift Detection Method) Detector
Monitors error rate and its standard deviation
"""

from typing import Dict, Optional

# Try to import from river, fall back to simple implementation
try:
    from river.drift import DDM
except (ImportError, AttributeError):
    from .simple_detectors import DDM


class DDMDetector:
    """
    DDM drift detector for federated learning
    """

    def __init__(self, warning_level: float = 2.0, drift_level: float = 3.0):
        """
        Initialize DDM detector

        Args:
            warning_level: Warning level threshold
            drift_level: Drift level threshold
        """
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.ddm = DDM(warning_level=warning_level, drift_level=drift_level)
        self.drift_detected = False
        self.warning_detected = False
        self.drift_timestamps = []
        self.warning_timestamps = []
        self.update_count = 0

    def update(self, value: float, timestamp: Optional[int] = None) -> bool:
        """
        Update detector with new binary value (0 = correct, 1 = error)

        Args:
            value: Binary value (0 or 1)
            timestamp: Optional timestamp

        Returns:
            bool: True if drift detected
        """
        self.update_count += 1
        if timestamp is None:
            timestamp = self.update_count

        # Update DDM
        self.ddm.update(int(value))

        # Check warning
        if self.ddm.warning_detected:
            self.warning_detected = True
            self.warning_timestamps.append(timestamp)

        # Check drift
        if self.ddm.drift_detected:
            self.drift_detected = True
            self.drift_timestamps.append(timestamp)
            return True

        return False

    def reset(self):
        """Reset the detector"""
        self.ddm = DDM(warning_level=self.warning_level, drift_level=self.drift_level)
        self.drift_detected = False
        self.warning_detected = False
        self.update_count = 0

    def get_drift_info(self) -> Dict:
        """Get information about detected drifts"""
        return {
            'detector': 'DDM',
            'drift_count': len(self.drift_timestamps),
            'drift_timestamps': self.drift_timestamps,
            'warning_count': len(self.warning_timestamps),
            'warning_timestamps': self.warning_timestamps,
            'total_updates': self.update_count,
            'parameters': {
                'warning_level': self.warning_level,
                'drift_level': self.drift_level
            }
        }


class FederatedDDM:
    """DDM detector adapted for federated learning"""

    def __init__(self, num_clients: int,
                 warning_level: float = 2.0,
                 drift_level: float = 3.0,
                 global_detector: bool = True):
        """Initialize Federated DDM"""
        self.num_clients = num_clients
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.global_detector_enabled = global_detector

        # Client-level detectors
        self.client_detectors = {
            i: DDMDetector(warning_level=warning_level, drift_level=drift_level)
            for i in range(num_clients)
        }

        # Global detector
        if global_detector:
            self.global_detector = DDMDetector(
                warning_level=warning_level,
                drift_level=drift_level
            )
        else:
            self.global_detector = None

    def update_client(self, client_id: int, error: int,
                     timestamp: Optional[int] = None) -> bool:
        """Update detector for a specific client"""
        if client_id not in self.client_detectors:
            return False

        return self.client_detectors[client_id].update(error, timestamp)

    def update_global(self, error: int,
                     timestamp: Optional[int] = None) -> bool:
        """Update global detector"""
        if self.global_detector is None:
            return False

        return self.global_detector.update(error, timestamp)

    def get_drifting_clients(self) -> list:
        """Get list of clients that detected drift"""
        return [
            client_id
            for client_id, detector in self.client_detectors.items()
            if detector.drift_detected
        ]

    def get_warning_clients(self) -> list:
        """Get list of clients in warning state"""
        return [
            client_id
            for client_id, detector in self.client_detectors.items()
            if detector.warning_detected
        ]

    def get_all_drift_info(self) -> Dict:
        """Get drift information for all clients"""
        info = {
            'detector': 'Federated-DDM',
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
