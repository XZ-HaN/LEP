import numpy as np
import random
import torch
import ase
from ase import Atoms
from ase.io.trajectory import Trajectory
from tqdm import tqdm
from typing import Tuple, List,Dict
import time
import sys
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import zlib
import base64
import pickle

class MonteCarloSampler:
    def __init__(
        self,
        model,
        analyzer,
        element_list: List[str],
        lattice_constant: float,
        max_steps: int = 1000,
        temperature: float = 1.0,
        swap_ratio: float = 0.5
    ):
        self.model = model
        self.analyzer = analyzer
        self.element_list = element_list
        self.lattice_constant = lattice_constant
        self.max_steps = max_steps
        self.temperature = temperature
        self.swap_ratio = swap_ratio
        self.device = next(model.parameters()).device

        # Initialize caches and neighbor table
        self.neighbor_table = None
        self.cached_features = None
        self.column_map = None
        self.atomic_energies = None
        self.total_energy = None
        
        # Element symbol ↔ atomic number mappings
        self.symbol_to_number = {}
        self.number_to_symbol = {}
        for el in element_list:
            num = ase.data.atomic_numbers[el]
            self.symbol_to_number[el] = num
            self.number_to_symbol[num] = el
        self.element_indices = defaultdict(list)
        self.index_to_element = {}
        self.feature_update_indices = {}
        
        
    def _initialize_element_indices(self, atoms):
        self.element_indices = defaultdict(list)
        self.index_to_element = {}
        
        numbers = atoms.numbers
        for idx, num in enumerate(numbers):
            symbol = self.number_to_symbol[num]
            self.element_indices[symbol].append(idx)
            self.index_to_element[idx] = symbol

    def _precompute_feature_update_indices(self, atoms):
        self.feature_update_indices = {}
        
        # Create (shell, element) → column index mapping
        self.shell_element_to_column = {}
        for col_name, col_idx in self.column_map.items():
            if '_' in col_name:
                shell_str, element = col_name.split('_', 1)
                if shell_str.isdigit():
                    self.shell_element_to_column[(int(shell_str), element)] = col_idx
        
        # Precompute update indices for each atom
        for atom_idx in range(len(atoms)):
            self.feature_update_indices[atom_idx] = []
            for shell_idx, neighbors in enumerate(self.neighbor_table[atom_idx]):
                for neighbor_idx in neighbors:
                    self.feature_update_indices[atom_idx].append((
                        neighbor_idx,
                        shell_idx
                    ))
                    
    def predict_energy(self, atoms: Atoms) -> float:
        if self.atomic_energies is None:
            with torch.no_grad():
                atomic_energies = self.model(
                    self.cached_features.unsqueeze(0),
                    torch.tensor([len(atoms)], device=self.device)
                )[0]
            self.atomic_energies = atomic_energies
            self.total_energy = atomic_energies.sum().item()
        return self.total_energy

    def _update_energy(self, modified_indices: List[int]):
        # Identify affected atoms (modified + neighbors)
        affected_indices = set(modified_indices)
        for idx in modified_indices:
            for neighbors in self.neighbor_table[idx]:
                affected_indices.update(neighbors)
        affected_indices = sorted(affected_indices)

        indices_tensor = torch.tensor(
            affected_indices, 
            dtype=torch.long, 
            device=self.device
        )

        old_energies = self.atomic_energies[indices_tensor].clone()

        affected_features = self.cached_features[indices_tensor].unsqueeze(0)

        with torch.no_grad():
            if affected_features.is_cuda and torch.cuda.is_bf16_supported():
                with torch.cuda.amp.autocast():
                    new_energies = self.model(
                        affected_features.half(),
                        torch.tensor([len(affected_indices)], device=self.device)
                    )[0].float()
            else:
                new_energies = self.model(
                    affected_features,
                    torch.tensor([len(affected_indices)], device=self.device)
                )[0]

        self.atomic_energies[indices_tensor] = new_energies

        delta = new_energies.sum() - old_energies.sum()
        self.total_energy += delta.item()

        return self.total_energy

    def _build_neighbor_table(self, atoms):
        neighbor_layers = self.analyzer._get_neighbors(atoms)
        n_layers = len(neighbor_layers) - 1
        neighbor_table = {}
        for i in range(len(atoms)):
            neighbor_table[i] = []
            neighbor_table[i].insert(0, [i])
            total_counter = Counter({i: 1})
            for layer in range(n_layers):
                current_shell = neighbor_layers[layer + 1][i]
                current_counter = Counter(current_shell)
                delta_counter = current_counter - total_counter
                delta_neighbors = []
                for idx in current_shell:
                    if delta_counter[idx] > 0:
                        delta_neighbors.append(idx)
                        delta_counter[idx] -= 1
                neighbor_table[i].append(delta_neighbors)
                total_counter = current_counter
        return neighbor_table

    def _initialize_cached_features(self, atoms):
        t0 = time.time()
        df = self.analyzer.create_atomic_environment_df(atoms,PBC_layers=1)
        self.column_map = {
            col: idx for idx, col in enumerate(df.columns)
        }
        self.cached_features = torch.tensor(df.values, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            atomic_energies = self.model(
                self.cached_features.unsqueeze(0),
                torch.tensor([len(atoms)], device=self.device)
            )[0]
        self.atomic_energies = atomic_energies
        self.total_energy = atomic_energies.sum().item()
        t1 = time.time()
        print(f'Feature initialization time: {t1-t0:.3f}s')
        
    def _update_cached_features(self, modified_indices: List[int], old_symbols: List[str], new_symbols: List[str]):
        update_ops = []

        for i, atom_idx in enumerate(modified_indices):
            old_symbol = old_symbols[i]
            new_symbol = new_symbols[i]

            for neighbor_idx, shell_idx in self.feature_update_indices.get(atom_idx, []):
                if (shell_idx, old_symbol) in self.shell_element_to_column:
                    col_old = self.shell_element_to_column[(shell_idx, old_symbol)]
                    update_ops.append((neighbor_idx, col_old, -1))

                if (shell_idx, new_symbol) in self.shell_element_to_column:
                    col_new = self.shell_element_to_column[(shell_idx, new_symbol)]
                    update_ops.append((neighbor_idx, col_new, +1))

        if not update_ops:
            return

        rows, cols, values = zip(*update_ops)
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_tensor = torch.tensor(cols, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.cached_features.index_put_(
            (rows_tensor, cols_tensor),
            values_tensor,
            accumulate=True
        )

    def random_replace(self, atoms: Atoms) -> Tuple[Atoms, List[int], str, str]:
        all_indices = list(range(len(atoms)))
        idx = random.choice(all_indices)

        old_symbol = self.index_to_element[idx]

        possible_symbols = [el for el in self.element_list if el != old_symbol]
        new_symbol = random.choice(possible_symbols)

        new_numbers = atoms.numbers.copy()
        new_numbers[idx] = self.symbol_to_number[new_symbol]

        new_atoms = Atoms(
            numbers=new_numbers,
            positions=atoms.positions.copy(),
            cell=atoms.cell.copy(),
            pbc=atoms.pbc
        )

        return new_atoms, [idx], old_symbol, new_symbol


    def random_swap(self, atoms: Atoms) -> Tuple[Atoms, List[int], List[str], List[str]]:
        available_elements = [el for el in self.element_indices if self.element_indices[el]]

        if len(available_elements) < 2:
            raise ValueError("At least two different elements required for swapping")

        etype1, etype2 = random.sample(available_elements, 2)

        idx1 = random.choice(self.element_indices[etype1])
        idx2 = random.choice(self.element_indices[etype2])

        old1 = self.index_to_element[idx1]
        old2 = self.index_to_element[idx2]

        new_numbers = atoms.numbers.copy()
        new_numbers[idx1], new_numbers[idx2] = new_numbers[idx2], new_numbers[idx1]

        new_atoms = Atoms(
            numbers=new_numbers,
            positions=atoms.positions.copy(),
            cell=atoms.cell.copy(),
            pbc=atoms.pbc
        )

        new_symbols = [self.number_to_symbol[new_numbers[idx1]], 
                       self.number_to_symbol[new_numbers[idx2]]]

        return new_atoms, [idx1, idx2], [old1, old2], new_symbols
    
    def _load_cached_features_from_atoms(self, atoms: Atoms) -> bool:
        if 'cached_features_compressed' not in atoms.info:
            return False

        try:
            b64_encoded = atoms.info['cached_features_compressed']
            compressed = base64.b64decode(b64_encoded)
            decompressed = zlib.decompress(compressed)
            cached_features_np = pickle.loads(decompressed)

            if 'cached_features_shape' in atoms.info:
                cached_features_np = cached_features_np.reshape(atoms.info['cached_features_shape'])
            if 'cached_features_dtype' in atoms.info:
                cached_features_np = cached_features_np.astype(np.dtype(atoms.info['cached_features_dtype']))

            self.cached_features = torch.tensor(
                cached_features_np, 
                dtype=torch.float32,
                device=self.device
            )

            if 'column_names' in atoms.info:
                column_names = atoms.info['column_names']
                self.column_map = {col: idx for idx, col in enumerate(column_names)}

            if 'neighbor_table' in atoms.info:
                serialized_table = atoms.info['neighbor_table']
                self.neighbor_table = {}
                for i_str, layers in serialized_table.items():
                    i = int(i_str)
                    self.neighbor_table[i] = [list(layer) for layer in layers]

            if 'atomic_energies_compressed' in atoms.info:
                b64_encoded = atoms.info['atomic_energies_compressed']
                compressed = base64.b64decode(b64_encoded)
                decompressed = zlib.decompress(compressed)
                atomic_energies_np = pickle.loads(decompressed)

                if 'atomic_energies_shape' in atoms.info:
                    atomic_energies_np = atomic_energies_np.reshape(atoms.info['atomic_energies_shape'])
                if 'atomic_energies_dtype' in atoms.info:
                    atomic_energies_np = atomic_energies_np.astype(np.dtype(atoms.info['atomic_energies_dtype']))

                self.atomic_energies = torch.tensor(
                    atomic_energies_np,
                    dtype=torch.float32,
                    device=self.device
                )

            return True

        except Exception as e:
            print(f"Cache load failed: {e}")
            return False

    def run_monte_carlo(
        self,
        initial_structure: Atoms,
        save_interval: int = 100,
        trajectory_file: str = 'accepted.traj',
        log_file: str = 'full_log.txt',
        resume: bool = False,
        val1000 = False
    ) -> List[dict]:
        current_structure = initial_structure.copy()
        
        cached_loaded = False
        if resume:
            try:
                from ase.io import read
                traj = Trajectory(trajectory_file, 'r')
                if len(traj) == 0:
                    raise ValueError("Empty trajectory file")

                last_atoms = read(trajectory_file, index=-1)

                cached_loaded = self._load_cached_features_from_atoms(last_atoms)
                if cached_loaded:
                    print("Successfully loaded cached environment data")
                    current_structure = last_atoms
                else:
                    print("Failed to load cached environment data")
                
                def get_last_step(log_file):
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            if not lines or len(lines) < 2:
                                return -1
                            last_line = lines[-1]
                            parts = last_line.strip().split('\t')
                            if len(parts) < 1:
                                return -1
                            return int(parts[0])
                    except (FileNotFoundError, IndexError, ValueError) as e:
                        return -1
                
                start_step = get_last_step(log_file) + 1
                if start_step < 0:
                    start_step = 0
            
            except Exception as e:
                print(f"Resume failed: {e}")
                start_step = 0
        else:
            start_step = 0

        if not cached_loaded or self.neighbor_table is None:
            self.neighbor_table = self._build_neighbor_table(current_structure)
        
        if not cached_loaded or self.cached_features is None:
            self._initialize_cached_features(current_structure)
            
        self._precompute_feature_update_indices(current_structure)
        
        if resume and cached_loaded:
            if 'energy' not in current_structure.info:
                print("Warning: Trajectory structure missing energy, recalculating")
                current_energy = self.predict_energy(current_structure)
            else:
                current_energy = current_structure.info['energy']
            
            if hasattr(self, 'atomic_energies') and self.atomic_energies is not None:
                self.total_energy = self.atomic_energies.sum().item()
                if abs(self.total_energy - current_energy) > 1e-4:
                    print(f"Warning: Atomic energy ({self.total_energy:.5f}) differs from structure energy ({current_energy:.5f}), using structure energy")
                    current_energy = self.total_energy
        else:
            current_energy = self.predict_energy(current_structure)

        k_B = 8.617e-5
        T_eV = k_B * self.temperature

        write_mode = 'a' if resume else 'w'

        if not resume:
            with open(log_file, 'w') as f:
                pass

        self._initialize_element_indices(current_structure)
        
        with Trajectory(trajectory_file, mode=write_mode) as traj:
            if not resume:
                current_structure.info['energy'] = current_energy
                self._save_cached_features_to_atoms(current_structure)
                traj.write(current_structure)

            results = []
            progression_name = f"Monte Carlo at {self.temperature}K"
            for step in tqdm(range(start_step, start_step + self.max_steps), desc=progression_name, file=sys.stdout):

                if random.random() < self.swap_ratio:
                    new_structure, modified_indices, old_symbols, new_symbols = self.random_swap(current_structure)
                else:
                    new_structure, modified_indices, old_symbol, new_symbol = self.random_replace(current_structure)
                    old_symbols = [old_symbol]
                    new_symbols = [new_symbol]
                

                cached_features_backup = self.cached_features.clone()
                atomic_energies_backup = self.atomic_energies.clone()
                current_energy = self.total_energy
                
                self._update_cached_features(modified_indices, old_symbols, new_symbols)

                new_energy = self._update_energy(modified_indices)
                delta_energy = new_energy - current_energy

                metropolis_prob = 1.0 if delta_energy <= 0 else np.exp(-delta_energy / T_eV)
                is_accepted = delta_energy <= 0 or random.random() < metropolis_prob

                step_info = {
                    'step': step,
                    'attempted_energy': new_energy,
                    'current_energy': current_energy,
                    'delta_energy': delta_energy,
                    'metropolis_prob': metropolis_prob,
                    'accepted': is_accepted
                }
                results.append(step_info)

                self.save_results(step_info, output_log=log_file)

                if is_accepted:
                    for i, idx in enumerate(modified_indices):
                        old_sym = old_symbols[i]
                        new_sym = new_symbols[i]

                        if idx in self.element_indices[old_sym]:
                            self.element_indices[old_sym].remove(idx)

                        self.element_indices[new_sym].append(idx)
                        
                        self.index_to_element[idx] = new_sym
                    current_structure = new_structure.copy()

                    current_energy = new_energy           
                    
                else:
                    self.cached_features = cached_features_backup
                    self.atomic_energies = atomic_energies_backup
                    self.total_energy = current_energy

                if step == 1000 and val1000 == True:
                    df = self.analyzer.create_atomic_environment_df(current_structure,PBC_layers=1)
                    current_features = torch.tensor(df.values, dtype=torch.float32).to(self.device)

                    if set(df.columns) != set(self.column_map.keys()):
                        raise ValueError("Column mismatch between cached and recalculated features.")

                    if not torch.allclose(self.cached_features, current_features, atol=1e-8):
                        raise ValueError("Feature validation failed at step 1000")     

                if (step+1) % save_interval == 0 and step != start_step:
                    current_structure.info['energy'] = current_energy
                    if step + 1 == self.max_steps + start_step - 1:
                        self._save_cached_features_to_atoms(current_structure)
                    traj.write(current_structure)

        return results

    def _save_cached_features_to_atoms(self, atoms: Atoms):
        for key in [
            'cached_features', 'column_names', 'neighbor_table', 
            'atomic_energies', 'cached_features_compressed'
        ]:
            if key in atoms.info:
                del atoms.info[key]

        if self.cached_features is not None:
            cached_features_np = self.cached_features.cpu().numpy()

            compressed = zlib.compress(pickle.dumps(cached_features_np))
            b64_encoded = base64.b64encode(compressed).decode('ascii')

            atoms.info['cached_features_compressed'] = b64_encoded
            atoms.info['cached_features_shape'] = cached_features_np.shape
            atoms.info['cached_features_dtype'] = str(cached_features_np.dtype)

        if self.column_map is not None:
            column_names = list(self.column_map.keys())
            atoms.info['column_names'] = column_names

        if self.neighbor_table is not None:
            serializable_table = {}
            for i, layers in self.neighbor_table.items():
                serializable_table[str(i)] = [list(layer) for layer in layers]
            atoms.info['neighbor_table'] = serializable_table

        if hasattr(self, 'atomic_energies') and self.atomic_energies is not None:
            atomic_energies_np = self.atomic_energies.cpu().numpy()
            if atomic_energies_np.dtype == np.float64:
                atomic_energies_np = atomic_energies_np.astype(np.float32)

            compressed = zlib.compress(pickle.dumps(atomic_energies_np))
            b64_encoded = base64.b64encode(compressed).decode('ascii')
            atoms.info['atomic_energies_compressed'] = b64_encoded
            atoms.info['atomic_energies_shape'] = atomic_energies_np.shape
            atoms.info['atomic_energies_dtype'] = str(atomic_energies_np.dtype)
            
    def save_results(self, step_info: dict, output_log: str = 'full_log.txt'):
        with open(output_log, 'a') as f:
            if f.tell() == 0:
                f.write("Step\tAttempted_E\tCurrent_E\tDelta_E\tP_accept\tAccepted\n")
            f.write(f"{step_info['step']}\t"
                    f"{step_info['attempted_energy']:.6f}\t"
                    f"{step_info['current_energy']:.6f}\t"
                    f"{step_info['delta_energy']:.6f}\t"
                    f"{step_info['metropolis_prob']:.6f}\t"
                    f"{int(step_info['accepted'])}\n")

def read_enhanced_log_file(log_file: str) -> Dict[str, np.ndarray]:
    data = defaultdict(list)
    with open(log_file, 'r') as f:
        headers = next(f).strip().split('\t')
        for line in f:
            values = line.strip().split('\t')
            for header, value in zip(headers, values):
                if header == 'Step':
                    data[header].append(int(value))
                elif header == 'Accepted':
                    data[header].append(bool(int(value)))
                else:
                    data[header].append(float(value))
    
    return {k: np.array(v) for k, v in data.items()}

def calculate_running_acceptance(steps: np.ndarray, accepted: np.ndarray, 
                                window_size: int = 100) -> np.ndarray:
    running_acceptance = np.zeros_like(steps, dtype=float)
    for i in range(len(steps)):
        start = max(0, i - window_size + 1)
        running_acceptance[i] = np.mean(accepted[start:i+1])
    return running_acceptance

def plot_energy(log_data):
    window_size = min(500, len(log_data['Step'])//10)
    running_acceptance = calculate_running_acceptance(
        log_data['Step'], 
        log_data['Accepted'],
        window_size=window_size
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    color_energy = 'tab:blue'
    ax1.set_ylabel('Energy (eV)', color=color_energy)
    ax1.plot(log_data['Step'], log_data['Current_E'], 
            label='Energy', color=color_energy, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_energy)

    color_acceptance = 'tab:purple'
    ax2.set_xlabel('MC Step')
    ax2.set_ylabel('Acceptance Rate', color=color_acceptance)
    ax2.plot(log_data['Step'], running_acceptance, 
            color=color_acceptance, label='Acceptance Rate')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color_acceptance)
    ax2.set_ylim(0, 1)

    plt.suptitle('Monte Carlo Simulation: Energy Evolution and Acceptance Rate')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()