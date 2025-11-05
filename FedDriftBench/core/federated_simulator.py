"""
Federated Learning Simulator for Drift Detection
Simulates a federated learning environment with multiple clients
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional
from copy import deepcopy


class SimpleNN(nn.Module):
    """Simple Neural Network for MNIST"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FederatedClient:
    """Represents a single client in federated learning"""

    def __init__(self, client_id: int, model: nn.Module, learning_rate: float = 0.01):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.data_buffer = []
        self.drift_detected = False
        self.performance_history = []

    def train_on_batch(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1) -> Dict:
        """Train the local model on a batch of data"""
        self.model.train()
        total_loss = 0.0

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        accuracy = self._compute_accuracy(X, y)
        self.performance_history.append(accuracy)

        return {
            'loss': total_loss / epochs,
            'accuracy': accuracy,
            'samples': len(y)
        }

    def _compute_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy on given data"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y).sum().item()
            return correct / len(y)

    def get_model_parameters(self) -> List[torch.Tensor]:
        """Get current model parameters"""
        return [param.data.clone() for param in self.model.parameters()]

    def set_model_parameters(self, parameters: List[torch.Tensor]):
        """Set model parameters"""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = new_param.clone()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict:
        """Evaluate the model on test data"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y).sum().item()
            accuracy = correct / len(y)

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(y)
        }


class FederatedServer:
    """Central server for federated learning"""

    def __init__(self, model: nn.Module, num_clients: int,
                 aggregation: str = 'fedavg'):
        self.global_model = model
        self.num_clients = num_clients
        self.aggregation = aggregation
        self.clients: List[FederatedClient] = []
        self.round = 0
        self.global_performance_history = []

        # Initialize clients
        for i in range(num_clients):
            client = FederatedClient(client_id=i, model=self.global_model)
            self.clients.append(client)

    def federated_averaging(self, client_parameters: List[List[torch.Tensor]],
                          client_weights: List[float]) -> List[torch.Tensor]:
        """Perform FedAvg aggregation"""
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Average parameters
        averaged_params = []
        for param_idx in range(len(client_parameters[0])):
            weighted_sum = sum(
                w * params[param_idx]
                for w, params in zip(normalized_weights, client_parameters)
            )
            averaged_params.append(weighted_sum)

        return averaged_params

    def select_clients(self, participation_rate: float = 1.0) -> List[int]:
        """Select clients for this round"""
        num_selected = max(1, int(self.num_clients * participation_rate))
        return np.random.choice(self.num_clients, num_selected, replace=False).tolist()

    def train_round(self, client_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                   participation_rate: float = 1.0,
                   local_epochs: int = 1) -> Dict:
        """Execute one round of federated learning"""
        self.round += 1

        # Select clients
        selected_clients = self.select_clients(participation_rate)

        # Distribute global model to clients
        global_params = [param.data.clone() for param in self.global_model.parameters()]
        for client_id in selected_clients:
            self.clients[client_id].set_model_parameters(global_params)

        # Local training
        client_params = []
        client_weights = []
        client_metrics = {}

        for client_id in selected_clients:
            if client_id in client_data:
                X, y = client_data[client_id]
                metrics = self.clients[client_id].train_on_batch(X, y, epochs=local_epochs)

                client_params.append(self.clients[client_id].get_model_parameters())
                client_weights.append(metrics['samples'])
                client_metrics[client_id] = metrics

        # Aggregate models
        if client_params:
            aggregated_params = self.federated_averaging(client_params, client_weights)

            # Update global model
            for param, new_param in zip(self.global_model.parameters(), aggregated_params):
                param.data = new_param.clone()

        # Compute average metrics
        avg_loss = np.mean([m['loss'] for m in client_metrics.values()])
        avg_accuracy = np.mean([m['accuracy'] for m in client_metrics.values()])

        round_result = {
            'round': self.round,
            'selected_clients': selected_clients,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'client_metrics': client_metrics
        }

        self.global_performance_history.append(avg_accuracy)

        return round_result

    def evaluate_global_model(self, test_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Evaluate global model on test data"""
        self.global_model.eval()
        X_test, y_test = test_data

        with torch.no_grad():
            outputs = self.global_model(X_test)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y_test)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_test).sum().item()
            accuracy = correct / len(y_test)

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'samples': len(y_test)
        }
