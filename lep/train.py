# structure_analysis.py

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional


# ==================== ✅ Configuration Section (User Modifiable) ====================
class Config:
    # Structural analysis parameters
    ELEMENT_LIST = ['Ti', 'Al', 'Nb', 'Mo', 'Zr']
    LATTICE_CONSTANT = 3.31
    NEIGHBOR_LAYERS = 6  # Max neighbor layers to analyze

    # Neural network parameters
    HIDDEN_SIZES = [500, 500, 500]  # Hidden layer structure
    ACTIVATION = 'relu'      # Activation function
    DROPOUT_RATE = 0.0       # Dropout rate

    # Training parameters
    TRAIN_RATIO = 0.8
    BATCH_SIZE = 16
    LR = 0.0005
    LOSS_FN = 'mse'          # Loss function type
    OPTIMIZER = 'adam'       # Optimizer type
    SCHEDULER = 'step_lr'    # Learning rate scheduler
    DEVICE = 'cuda'          # Device (cuda or cpu)

    # Other configurations
    MAX_EPOCHS = 100
    VERBOSE_TRAIN = True
    
class StructureAnalyzer:
    def __init__(
        self,
        element_list: List[str],
        lattice_constant: float,
        n_layers: int,
        lattice_type = 'BCC',
        neighbor_distances: Optional[List[float]] = None
    ):
        self.element_list = element_list
        self.lattice_constant = lattice_constant
        self.n_layers = n_layers

        if neighbor_distances is None:
            if lattice_type == 'BCC':
                self.crystal_nl = [0, 0.866, 1.0, 1.414, 1.658, 1.732, 2.0, 2.179, 2.236,
                              2.449, 2.598, 2.828, 2.958, 3.0, 3.162, 3.279, 3.317,
                              3.464, 3.571, 3.606, 3.742, 3.841, 4.0, 4.093, 4.123,
                              4.243, 4.33, 4.359, 4.472, 4.555, 4.583, 4.69, 4.770,
                              4.899, 4.975, 5.0]
                
            elif lattice_type == 'FCC':
                self.crystal_nl = [0, 0.707, 1.0, 1.225, 1.414, 1.581, 1.732, 1.871, 2.0,
                                   2.121, 2.236, 2.345, 2.449, 2.550, 2.739, 2.828, 2.915,
                                   3.0, 3.082, 3.162, 3.317, 3.391, 3.464, 3.535, 3.606,
                                   3.674, 3.742, 4.0, 4.062, 4.123, 4.243, 4.301, 4.359,
                                   4.472, 4.528, 4.583, 4.690, 4.899, 4.950, 5.0]
                
            elif lattice_type == 'HCP':
                self.crystal_nl = [1.0, 1.633, 1.732, 1.915, 2.0, 2.291, 2.309, 2.517, 2.646, 2.828, 
                                   3.0, 3.058, 3.214, 3.317, 3.464, 3.512, 3.633, 3.742, 3.858, 4.0, 
                                   4.041, 4.163, 4.359, 4.388, 4.472, 4.619, 4.714, 4.799, 4.899, 4.959, 5.0]
                
        else:
            self.crystal_nl = neighbor_distances

    @property
    def input_dim(self):
        """Automatically calculate input dimension: (layers + 1) * element types"""
        return (self.n_layers + 1) * len(self.element_list)

    def _get_periodic_images(self, positions, cell, PBC_layers=2):
        images = []
        for x in range(-PBC_layers, PBC_layers + 1):
            for y in range(-PBC_layers, PBC_layers + 1):
                for z in range(-PBC_layers, PBC_layers + 1):
                    shift = x * cell[0] + y * cell[1] + z * cell[2]
                    images.append(positions + shift)
        return np.vstack(images)

    def _get_neighbors(self, atoms, PBC_layers=2):
        neighbor_results = []
        positions = atoms.get_positions()
        cell = atoms.get_cell().array
        extended_positions = self._get_periodic_images(positions, cell, PBC_layers=PBC_layers)

        tree = cKDTree(extended_positions)

        for layer in range(1, self.n_layers + 2):
            cutoff_radius = self.lattice_constant * self.crystal_nl[layer - 1] + 0.005
            layer_neighbors_dict = defaultdict(list)
            for i, pos in enumerate(positions):
                indices = tree.query_ball_point(pos, r=cutoff_radius)
                indices = [idx % len(positions) for idx in indices]
                layer_neighbors_dict[i].extend(indices)
            neighbor_results.append(layer_neighbors_dict)
        return neighbor_results

    def create_atomic_environment_df(self, atoms, PBC_layers=2):
        neighbor_layers = self._get_neighbors(atoms, PBC_layers=PBC_layers)
        results = []

        for i, atom in enumerate(atoms):
            atom_info = {f"0_{el}": 0 for el in self.element_list}
            previous_neighbors = []

            for layer in range(self.n_layers + 1):
                current_neighbors = neighbor_layers[layer][i]
                unique_neighbors = Counter(current_neighbors) - Counter(previous_neighbors)
                previous_neighbors = current_neighbors

                for j in unique_neighbors:
                    symbol = atoms[j].symbol
                    if symbol in self.element_list:
                        key = f"{layer}_{symbol}"
                        atom_info[key] = atom_info.get(key, 0) + unique_neighbors[j]

            results.append(atom_info)

        columns = [f"{i}_{el}" for i in range(self.n_layers + 1) for el in self.element_list]
        return pd.DataFrame(results, columns=columns).fillna(0)

class NNmodel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str,
        dropout_rate: float,
        output_size: int = 1,
        per_atom_output: bool = True
    ):
        super().__init__()
        self.activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation_fn = self.activations[activation.lower()]
        self.dropout = nn.Dropout(dropout_rate)
        self.per_atom_output = per_atom_output

        layers = []
        prev_size = input_size
        for i in range(len(hidden_sizes)):
            size = hidden_sizes[i]
            layers.append(nn.Linear(prev_size, size))
            layers.append(self.activation_fn)
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, lengths):
        max_length = x.size(1)
        mask = torch.arange(max_length).expand(len(lengths), max_length).to(x.device) < lengths.unsqueeze(1)
        
        x = self.fc(x)
        
        if self.per_atom_output:
            return x.squeeze(-1) * mask.float()
        else:
            x = x * mask.unsqueeze(2)
            return x.sum(1)
    
class EnergyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        return x, torch.tensor(y, dtype=torch.float32)


class EnergyTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_ratio: float,
        batch_size: int,
        lr: float,
        loss_fn: str,
        optimizer_name: str,
        scheduler_name: Optional[str],
        device: str
    ):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'mae':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError("loss_fn must be 'mse' or 'mae'")

        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            raise ValueError("optimizer_name must be 'adam' or 'sgd'")

        self.scheduler = None
        if scheduler_name == 'step_lr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=5, verbose=True
            )

    def _pad_sequence(self, seq, max_length):
        return F.pad(seq, (0, 0, 0, max_length - seq.size(0)))

    def _collate_fn(self, batch):
        x, y = zip(*batch)
        return list(x), torch.tensor(y, dtype=torch.float32)

    def prepare_data(self, dataset,seed=0):
        generator = torch.Generator().manual_seed(seed)
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for x, y in self.train_loader:
            lengths = torch.tensor([seq.size(0) for seq in x]).to(self.device)
            max_len = max(lengths)

            x_padded = torch.stack([self._pad_sequence(seq, max_len) for seq in x]).to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x_padded, lengths)
            if self.model.per_atom_output:
                total_energies = outputs.sum(dim=1)
                average_outputs = total_energies / lengths.float()
            else:
                average_outputs = outputs.squeeze() / lengths.float()
            loss = self.loss_fn(average_outputs.squeeze(), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR):
            self.scheduler.step()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                lengths = torch.tensor([seq.size(0) for seq in x]).to(self.device)
                max_len = max(lengths)

                x_padded = torch.stack([self._pad_sequence(seq, max_len) for seq in x]).to(self.device)
                y = y.to(self.device)

                outputs = self.model(x_padded, lengths)
                if self.model.per_atom_output:
                    total_energies = outputs.sum(dim=1)
                    average_outputs = total_energies / lengths.float()
                else:
                    average_outputs = outputs.squeeze() / lengths.float()
                loss = self.loss_fn(average_outputs.squeeze(), y)
                total_loss += loss.item()

        if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(total_loss)

        return total_loss / len(self.val_loader)

    def train(self, epochs=100, verbose=True):
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4e} | "
                      f"Val Loss: {val_loss:.4e}")

        return train_losses, val_losses

    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Progress')
        plt.show()

    def evaluate(self, full_dataset):
        full_loader = DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn
        )

        self.model.eval()
        actual, predicted = [], []

        with torch.no_grad():
            for x, y in full_loader:
                lengths = torch.tensor([seq.size(0) for seq in x]).to(self.device)
                max_len = max(lengths)

                x_padded = torch.stack([self._pad_sequence(seq, max_len) for seq in x]).to(self.device)
                y = y.to(self.device)

                outputs = self.model(x_padded, lengths)
                if self.model.per_atom_output:
                    total_energies = outputs.sum(dim=1)
                    average_outputs = total_energies / lengths.float()
                else:
                    average_outputs = outputs.squeeze() / lengths.float()
                actual.extend(y.cpu().numpy())
                predicted.extend(average_outputs.squeeze().cpu().numpy())

        return np.array(actual), np.array(predicted)
    

    def plot_predictions(self, actual, predicted):
        rmse = np.sqrt(np.mean((actual - predicted)**2))

        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)],
                'r--', lw=2)
        plt.text(0.05, 0.95, f'RMSE = {rmse:.4f} eV/atom',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        plt.xlabel('DFT Energy (eV/atom)')
        plt.ylabel('Predicted Energy (eV/atom)')
        plt.show()
        
    def evaluate_val(self):
        self.model.eval()
        actual, predicted = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                lengths = torch.tensor([seq.size(0) for seq in x]).to(self.device)
                max_len = max(lengths)

                x_padded = torch.stack([self._pad_sequence(seq, max_len) for seq in x]).to(self.device)
                y = y.to(self.device)

                outputs = self.model(x_padded, lengths)
                if self.model.per_atom_output:
                    total_energies = outputs.sum(dim=1)
                    average_outputs = total_energies / lengths.float()
                else:
                    average_outputs = outputs.squeeze() / lengths.float()
                actual.extend(y.cpu().numpy())
                predicted.extend(average_outputs.squeeze().cpu().numpy())
        
        actual, predicted = np.array(actual), np.array(predicted)

        rmse = np.sqrt(np.mean((actual - predicted)**2))

        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)],
                'r--', lw=2)
        plt.text(0.05, 0.95, f'RMSE = {rmse:.4f} eV/atom',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        plt.title('Result Validation', fontsize=14)
        plt.xlabel('DFT Energy (eV/atom)')
        plt.ylabel('Predicted Energy (eV/atom)')
        plt.show()