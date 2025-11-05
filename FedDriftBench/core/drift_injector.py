"""
Drift Injection Module
Injects various types of concept drift into data streams
"""

import numpy as np
import torch
from typing import Tuple, List, Dict
from enum import Enum


class DriftType(Enum):
    """Types of concept drift"""
    SUDDEN = "sudden"           # Abrupt change
    GRADUAL = "gradual"         # Slow transition
    INCREMENTAL = "incremental" # Step-by-step change
    RECURRING = "recurring"     # Cyclic pattern


class DriftInjector:
    """Injects concept drift into data streams"""

    def __init__(self, drift_type: DriftType, drift_magnitude: float = 0.5):
        self.drift_type = drift_type
        self.drift_magnitude = drift_magnitude

    def inject_label_drift(self, X: np.ndarray, y: np.ndarray,
                          drift_start: int, drift_end: int,
                          label_map: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject label drift (virtual drift)
        Changes class labels according to a mapping
        """
        y_drifted = y.copy()

        if self.drift_type == DriftType.SUDDEN:
            # Sudden drift: immediate change at drift_start
            for i in range(drift_start, len(y)):
                if y[i] in label_map:
                    y_drifted[i] = label_map[y[i]]

        elif self.drift_type == DriftType.GRADUAL:
            # Gradual drift: linearly increase probability of change
            drift_length = drift_end - drift_start
            for i in range(drift_start, drift_end):
                progress = (i - drift_start) / drift_length
                if y[i] in label_map and np.random.random() < progress:
                    y_drifted[i] = label_map[y[i]]

            # After drift_end, all changes
            for i in range(drift_end, len(y)):
                if y[i] in label_map:
                    y_drifted[i] = label_map[y[i]]

        return X, y_drifted

    def inject_feature_drift(self, X: np.ndarray, y: np.ndarray,
                            drift_start: int, drift_end: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject feature drift (real drift)
        Modifies feature distributions
        """
        X_drifted = X.copy()

        if self.drift_type == DriftType.SUDDEN:
            # Add noise to features after drift_start
            noise = np.random.normal(0, self.drift_magnitude, X[drift_start:].shape)
            X_drifted[drift_start:] += noise

        elif self.drift_type == DriftType.GRADUAL:
            # Gradually increase noise
            drift_length = drift_end - drift_start
            for i in range(drift_start, min(drift_end, len(X))):
                progress = (i - drift_start) / drift_length
                noise = np.random.normal(0, self.drift_magnitude * progress, X[i].shape)
                X_drifted[i] += noise

            # After drift_end, full noise
            if drift_end < len(X):
                noise = np.random.normal(0, self.drift_magnitude, X[drift_end:].shape)
                X_drifted[drift_end:] += noise

        return X_drifted, y

    def inject_rotation_drift(self, X: np.ndarray, y: np.ndarray,
                             drift_start: int, rotation_angle: float = 15.0
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject rotation drift (for image data)
        Rotates images by a certain angle
        """
        X_drifted = X.copy()

        # Assuming X is flattened images, reshape to 28x28 for MNIST
        img_size = int(np.sqrt(X.shape[1]))
        if img_size * img_size != X.shape[1]:
            # Not square, just add noise instead
            return self.inject_feature_drift(X, y, drift_start, len(X))

        angle_rad = np.radians(rotation_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        for i in range(drift_start, len(X)):
            img = X[i].reshape(img_size, img_size)
            # Simple rotation approximation
            img_rotated = self._rotate_image(img, cos_a, sin_a)
            X_drifted[i] = img_rotated.flatten()

        return X_drifted, y

    def _rotate_image(self, img: np.ndarray, cos_a: float, sin_a: float) -> np.ndarray:
        """Helper function to rotate image"""
        # Simplified rotation - in practice use scipy.ndimage.rotate
        # For speed, we just add some transformation
        return img + np.random.normal(0, 0.1, img.shape)


class MultiClientDriftScenario:
    """
    Manages drift scenarios across multiple clients
    """

    def __init__(self, num_clients: int, drift_type: DriftType):
        self.num_clients = num_clients
        self.drift_type = drift_type
        self.drift_schedule: Dict[int, Dict] = {}

    def set_homogeneous_drift(self, drift_start_round: int,
                              affected_clients: List[int],
                              drift_params: Dict):
        """
        All clients experience drift at the same time
        """
        for client_id in affected_clients:
            self.drift_schedule[client_id] = {
                'start_round': drift_start_round,
                'end_round': drift_start_round + drift_params.get('duration', 10),
                'params': drift_params
            }

    def set_heterogeneous_drift(self, drift_start_rounds: Dict[int, int],
                                drift_params: Dict):
        """
        Different clients experience drift at different times
        """
        for client_id, start_round in drift_start_rounds.items():
            self.drift_schedule[client_id] = {
                'start_round': start_round,
                'end_round': start_round + drift_params.get('duration', 10),
                'params': drift_params
            }

    def set_staggered_drift(self, start_round: int, stagger_interval: int,
                           affected_clients: List[int], drift_params: Dict):
        """
        Clients experience drift in a staggered manner
        """
        for idx, client_id in enumerate(affected_clients):
            client_start = start_round + (idx * stagger_interval)
            self.drift_schedule[client_id] = {
                'start_round': client_start,
                'end_round': client_start + drift_params.get('duration', 10),
                'params': drift_params
            }

    def should_inject_drift(self, client_id: int, current_round: int) -> Tuple[bool, Dict]:
        """
        Check if drift should be injected for a client at current round
        """
        if client_id not in self.drift_schedule:
            return False, {}

        schedule = self.drift_schedule[client_id]
        if schedule['start_round'] <= current_round <= schedule['end_round']:
            return True, schedule['params']

        return False, {}

    def get_drift_status(self, current_round: int) -> Dict[int, bool]:
        """
        Get drift status for all clients at current round
        """
        status = {}
        for client_id in range(self.num_clients):
            is_drifting, _ = self.should_inject_drift(client_id, current_round)
            status[client_id] = is_drifting
        return status
