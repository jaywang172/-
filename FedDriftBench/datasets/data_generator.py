"""
Data Generator for Federated Learning Experiments
Generates synthetic and real datasets partitioned across clients
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from sklearn.datasets import make_classification
from torchvision import datasets, transforms
import os


class SyntheticDataset:
    """Generate synthetic classification dataset"""

    def __init__(self, n_samples: int = 10000, n_features: int = 20,
                 n_classes: int = 2, n_informative: int = 15,
                 random_state: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_informative = n_informative
        self.random_state = random_state

        # Generate data
        self.X, self.y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            n_classes=n_classes,
            random_state=random_state
        )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full dataset"""
        return self.X, self.y


class MNISTFederated:
    """MNIST dataset partitioned for federated learning"""

    def __init__(self, num_clients: int = 10, data_dir: str = './data',
                 non_iid: bool = False, alpha: float = 0.5):
        """
        Initialize MNIST federated dataset

        Args:
            num_clients: Number of clients
            data_dir: Directory to store data
            non_iid: Whether to create non-IID partitions
            alpha: Dirichlet concentration parameter (lower = more non-IID)
        """
        self.num_clients = num_clients
        self.data_dir = data_dir
        self.non_iid = non_iid
        self.alpha = alpha

        # Load MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )

        # Partition data
        self.client_data = self._partition_data()

    def _partition_data(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Partition training data among clients"""
        train_data = []
        train_labels = []

        for data, label in self.train_dataset:
            train_data.append(data)
            train_labels.append(label)

        train_data = torch.stack(train_data)
        train_labels = torch.tensor(train_labels)

        if self.non_iid:
            client_data = self._non_iid_partition(train_data, train_labels)
        else:
            client_data = self._iid_partition(train_data, train_labels)

        return client_data

    def _iid_partition(self, data: torch.Tensor,
                      labels: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create IID partitions"""
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        split_indices = np.array_split(indices, self.num_clients)

        client_data = {}
        for client_id, client_indices in enumerate(split_indices):
            client_data[client_id] = {
                'X': data[client_indices],
                'y': labels[client_indices]
            }

        return client_data

    def _non_iid_partition(self, data: torch.Tensor,
                          labels: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create non-IID partitions using Dirichlet distribution"""
        n_classes = len(torch.unique(labels))
        label_indices = [torch.where(labels == i)[0].numpy() for i in range(n_classes)]

        client_data = {i: {'X': [], 'y': []} for i in range(self.num_clients)}

        # Use Dirichlet distribution to assign samples
        for class_indices in label_indices:
            np.random.shuffle(class_indices)
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            client_indices = np.split(class_indices, proportions)

            for client_id, indices in enumerate(client_indices):
                client_data[client_id]['X'].append(data[indices])
                client_data[client_id]['y'].append(labels[indices])

        # Concatenate data for each client
        for client_id in range(self.num_clients):
            if client_data[client_id]['X']:
                client_data[client_id]['X'] = torch.cat(client_data[client_id]['X'])
                client_data[client_id]['y'] = torch.cat(client_data[client_id]['y'])
            else:
                # Assign empty tensors if no data
                client_data[client_id]['X'] = torch.tensor([])
                client_data[client_id]['y'] = torch.tensor([])

        return client_data

    def get_client_data(self, client_id: int) -> Dict[str, torch.Tensor]:
        """Get data for a specific client"""
        return self.client_data[client_id]

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test dataset"""
        test_data = []
        test_labels = []

        for data, label in self.test_dataset:
            test_data.append(data)
            test_labels.append(label)

        return torch.stack(test_data), torch.tensor(test_labels)

    def create_data_stream(self, client_id: int,
                          batch_size: int = 32) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create a stream of batches for a client

        Args:
            client_id: Client ID
            batch_size: Batch size

        Returns:
            List of (X_batch, y_batch) tuples
        """
        client_data = self.client_data[client_id]
        X, y = client_data['X'], client_data['y']

        n_samples = len(X)
        n_batches = max(1, n_samples // batch_size)

        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batches.append((X[start_idx:end_idx], y[start_idx:end_idx]))

        return batches


class StreamDataGenerator:
    """Generate streaming data with concept drift"""

    def __init__(self, base_dataset: MNISTFederated, num_rounds: int):
        self.base_dataset = base_dataset
        self.num_rounds = num_rounds

    def generate_round_data(self, round_num: int, client_id: int,
                           samples_per_round: int = 100,
                           drift_applied: bool = False,
                           drift_magnitude: float = 0.0
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate data for a specific client in a specific round

        Args:
            round_num: Current round number
            client_id: Client ID
            samples_per_round: Number of samples per round
            drift_applied: Whether drift should be applied
            drift_magnitude: Magnitude of drift

        Returns:
            (X_batch, y_batch)
        """
        client_data = self.base_dataset.get_client_data(client_id)
        X_all, y_all = client_data['X'], client_data['y']

        # Get samples for this round
        total_samples = len(X_all)
        if total_samples == 0:
            return torch.tensor([]), torch.tensor([])

        start_idx = (round_num * samples_per_round) % total_samples
        end_idx = min(start_idx + samples_per_round, total_samples)

        X_batch = X_all[start_idx:end_idx]
        y_batch = y_all[start_idx:end_idx]

        # Apply drift if needed
        if drift_applied and drift_magnitude > 0:
            noise = torch.randn_like(X_batch) * drift_magnitude
            X_batch = X_batch + noise

        return X_batch, y_batch
