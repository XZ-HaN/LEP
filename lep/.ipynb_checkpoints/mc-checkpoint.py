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
        self.device = next(model.parameters()).device  # 获取模型设备

        # 初始化缓存和邻居表
        self.neighbor_table = None
        self.cached_features = None
        self.column_map = None  # 列名到索引的映射
        self.atomic_energies = None
        self.total_energy = None
        
        # 添加元素符号和原子序数的双向映射
        self.symbol_to_number = {}
        self.number_to_symbol = {}
        for el in element_list:
            num = ase.data.atomic_numbers[el]
            self.symbol_to_number[el] = num
            self.number_to_symbol[num] = el
        # 添加元素索引缓存
        self.element_indices = defaultdict(list)
        # 添加原子索引到元素的映射
        self.index_to_element = {}
        # 添加特征更新索引的缓存
        self.feature_update_indices = {}
        
        
    def _initialize_element_indices(self, atoms):
        """初始化元素索引缓存"""
        self.element_indices = defaultdict(list)
        self.index_to_element = {}
        
        numbers = atoms.numbers
        for idx, num in enumerate(numbers):
            symbol = self.number_to_symbol[num]
            self.element_indices[symbol].append(idx)
            self.index_to_element[idx] = symbol
    def _precompute_feature_update_indices(self, atoms):
        """预计算特征更新索引"""
        self.feature_update_indices = {}
        
        # 创建 (壳层, 元素) 到列索引的映射
        self.shell_element_to_column = {}
        for col_name, col_idx in self.column_map.items():
            if '_' in col_name:
                shell_str, element = col_name.split('_', 1)
                if shell_str.isdigit():
                    self.shell_element_to_column[(int(shell_str), element)] = col_idx
        
        # 为每个原子预计算其邻居的更新索引
        for atom_idx in range(len(atoms)):
            self.feature_update_indices[atom_idx] = []
            for shell_idx, neighbors in enumerate(self.neighbor_table[atom_idx]):
                for neighbor_idx in neighbors:
                    self.feature_update_indices[atom_idx].append((
                        neighbor_idx,  # 邻居索引
                        shell_idx      # 壳层索引
                    ))
                    
    def predict_energy(self, atoms: Atoms) -> float:
        """预测能量（支持原子级缓存）"""
        if self.atomic_energies is None:
            # 首次计算：完整预测并缓存
            with torch.no_grad():
                atomic_energies = self.model(
                    self.cached_features.unsqueeze(0),
                    torch.tensor([len(atoms)], device=self.device)
                )[0]
            self.atomic_energies = atomic_energies
            self.total_energy = atomic_energies.sum().item()
        return self.total_energy

    def _update_energy(self, modified_indices: List[int]):
        """增量更新受影响原子的能量 - GPU 优化版本"""
        # 1. 识别受影响原子（修改原子+其近邻）
        affected_indices = set(modified_indices)
        for idx in modified_indices:
            # 包括所有壳层的邻居
            for neighbors in self.neighbor_table[idx]:
                affected_indices.update(neighbors)
        affected_indices = sorted(affected_indices)

        # 直接在 GPU 上创建索引张量
        indices_tensor = torch.tensor(
            affected_indices, 
            dtype=torch.long, 
            device=self.device
        )

        # 2. 备份旧能量 - 使用索引直接从 GPU 张量中获取
        old_energies = self.atomic_energies[indices_tensor].clone()

        # 3. 提取受影响原子特征 - 使用 GPU 索引
        affected_features = self.cached_features[indices_tensor].unsqueeze(0)

        # 4. 预测新能量 - 优化推理过程
        with torch.no_grad():
            # 使用半精度加速推理（如果 GPU 支持）
            if affected_features.is_cuda and torch.cuda.is_bf16_supported():
                with torch.cuda.amp.autocast():
                    new_energies = self.model(
                        affected_features.half(),  # 使用半精度
                        torch.tensor([len(affected_indices)], device=self.device)
                    )[0].float()  # 转换回全精度
            else:
                new_energies = self.model(
                    affected_features,
                    torch.tensor([len(affected_indices)], device=self.device)
                )[0]

        # 5. 更新缓存 - 使用 GPU 索引更新
        self.atomic_energies[indices_tensor] = new_energies

        # 6. 计算能量变化 - 完全在 GPU 上计算
        delta = new_energies.sum() - old_energies.sum()
        self.total_energy += delta.item()

        return self.total_energy

    def _build_neighbor_table(self, atoms):
        neighbor_layers = self.analyzer._get_neighbors(atoms)
        n_layers = len(neighbor_layers) - 1  # 去掉第0层
        neighbor_table = {}
        for i in range(len(atoms)):
            neighbor_table[i] = []
            neighbor_table[i].insert(0, [i])  # 第0层保存自己
            total_counter = Counter({i: 1})
            for layer in range(n_layers):
                current_shell = neighbor_layers[layer + 1][i]  # layer+1 对应实际壳层
                # 计算当前壳层的计数器
                current_counter = Counter(current_shell)
                # 计算增量计数器（当前壳层 - 已处理计数器）
                delta_counter = current_counter - total_counter
                # 生成增量邻居列表（保留重复项）
                delta_neighbors = []
                for idx in current_shell:
                    if delta_counter[idx] > 0:
                        delta_neighbors.append(idx)
                        delta_counter[idx] -= 1
                neighbor_table[i].append(delta_neighbors)
                # 更新已处理计数器
                total_counter = current_counter
        return neighbor_table

    def _initialize_cached_features(self, atoms):
        t0 = time.time()
        """初始化缓存的环境张量"""
        df = self.analyzer.create_atomic_environment_df(atoms,PBC_layers=1)
        self.column_map = {
            col: idx for idx, col in enumerate(df.columns)
        }
        self.cached_features = torch.tensor(df.values, dtype=torch.float32).to(self.device)
        # 直接在 GPU 上创建张量
        self.cached_features = torch.tensor(
            df.values, 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 初始化原子能量缓存也在 GPU 上
        with torch.no_grad():
            atomic_energies = self.model(
                self.cached_features.unsqueeze(0),
                torch.tensor([len(atoms)], device=self.device)
            )[0]
        self.atomic_energies = atomic_energies
        self.total_energy = atomic_energies.sum().item()
        t1 = time.time()
        print(f'It has taken {t1-t0:.3f}s to initialize structure features.')
        
    def _update_cached_features(self, modified_indices: List[int], old_symbols: List[str], new_symbols: List[str]):
        # 创建更新操作列表 (索引, 列, 值)
        update_ops = []

        for i, atom_idx in enumerate(modified_indices):
            old_symbol = old_symbols[i]
            new_symbol = new_symbols[i]

            # 获取该原子的所有邻居更新索引
            for neighbor_idx, shell_idx in self.feature_update_indices.get(atom_idx, []):
                # 旧元素列索引
                if (shell_idx, old_symbol) in self.shell_element_to_column:
                    col_old = self.shell_element_to_column[(shell_idx, old_symbol)]
                    update_ops.append((neighbor_idx, col_old, -1))

                # 新元素列索引
                if (shell_idx, new_symbol) in self.shell_element_to_column:
                    col_new = self.shell_element_to_column[(shell_idx, new_symbol)]
                    update_ops.append((neighbor_idx, col_new, +1))

        # 如果没有更新操作，直接返回
        if not update_ops:
            return

        # 向量化更新
        rows, cols, values = zip(*update_ops)
        rows_tensor = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_tensor = torch.tensor(cols, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        # 一次性更新所有特征
        self.cached_features.index_put_(
            (rows_tensor, cols_tensor),
            values_tensor,
            accumulate=True  # 累加更新（而不是替换）
        )

    def random_replace(self, atoms: Atoms) -> Tuple[Atoms, List[int], str, str]:
        # 从所有原子中随机选择一个（使用缓存索引）
        all_indices = list(range(len(atoms)))
        idx = random.choice(all_indices)

        # 记录旧符号
        old_symbol = self.index_to_element[idx]

        # 选择新元素
        possible_symbols = [el for el in self.element_list if el != old_symbol]
        new_symbol = random.choice(possible_symbols)

        # 创建新原子对象
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

        # 使用预计算的缓存
        available_elements = [el for el in self.element_indices if self.element_indices[el]]

        if len(available_elements) < 2:
            raise ValueError("At least two different elements required for swapping")

        # 随机选择两种不同的元素
        etype1, etype2 = random.sample(available_elements, 2)

        # 从缓存中随机选择原子
        idx1 = random.choice(self.element_indices[etype1])
        idx2 = random.choice(self.element_indices[etype2])

        # 记录旧符号
        old1 = self.index_to_element[idx1]
        old2 = self.index_to_element[idx2]

        # 创建新原子对象
        new_numbers = atoms.numbers.copy()
        new_numbers[idx1], new_numbers[idx2] = new_numbers[idx2], new_numbers[idx1]

        new_atoms = Atoms(
            numbers=new_numbers,
            positions=atoms.positions.copy(),
            cell=atoms.cell.copy(),
            pbc=atoms.pbc
        )

        # 新符号（交换后）
        new_symbols = [self.number_to_symbol[new_numbers[idx1]], 
                       self.number_to_symbol[new_numbers[idx2]]]

        return new_atoms, [idx1, idx2], [old1, old2], new_symbols
    
    def _load_cached_features_from_atoms(self, atoms: Atoms) -> bool:
        """从 atoms.info 字典加载环境数据"""
        if 'cached_features_compressed' not in atoms.info:
            return False

        try:
            # 加载环境张量
            b64_encoded = atoms.info['cached_features_compressed']
            compressed = base64.b64decode(b64_encoded)
            decompressed = zlib.decompress(compressed)
            cached_features_np = pickle.loads(decompressed)

            # 确保正确形状和类型
            if 'cached_features_shape' in atoms.info:
                cached_features_np = cached_features_np.reshape(atoms.info['cached_features_shape'])
            if 'cached_features_dtype' in atoms.info:
                cached_features_np = cached_features_np.astype(np.dtype(atoms.info['cached_features_dtype']))

            self.cached_features = torch.tensor(
                cached_features_np, 
                dtype=torch.float32,
                device=self.device
            )

            # 加载列映射
            if 'column_names' in atoms.info:
                column_names = atoms.info['column_names']
                self.column_map = {col: idx for idx, col in enumerate(column_names)}

            # 加载邻居表
            if 'neighbor_table' in atoms.info:
                serialized_table = atoms.info['neighbor_table']
                self.neighbor_table = {}
                for i_str, layers in serialized_table.items():
                    i = int(i_str)
                    # 将每层邻居转换为列表
                    self.neighbor_table[i] = [list(layer) for layer in layers]

            # 加载原子能量
            if 'atomic_energies_compressed' in atoms.info:
                b64_encoded = atoms.info['atomic_energies_compressed']
                compressed = base64.b64decode(b64_encoded)
                decompressed = zlib.decompress(compressed)
                atomic_energies_np = pickle.loads(decompressed)

                # 确保正确形状和类型
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
            print(f"加载缓存数据失败: {e}")
            return False

    def run_monte_carlo(
        self,
        initial_structure: Atoms,
        save_interval: int = 100,
        trajectory_file: str = 'accepted.traj',
        log_file: str = 'full_log.txt',
        resume: bool = False,  # 新增参数
        val1000 = False
    ) -> List[dict]:

        # 初始化结构
        current_structure = initial_structure.copy()
        
        # 如果 resume=True，从轨迹文件读取最后一步的结构作为初始结构        
        cached_loaded = False
        if resume:
            try:
                # 使用 ASE 的 read 函数确保加载所有数据
                from ase.io import read
                traj = Trajectory(trajectory_file, 'r')
                if len(traj) == 0:
                    raise ValueError("轨迹文件为空，无法恢复")

                # 直接读取最后一个结构（避免 Trajectory 对象的限制）
                last_atoms = read(trajectory_file, index=-1)

                # 尝试加载缓存数据
                cached_loaded = self._load_cached_features_from_atoms(last_atoms)
                if cached_loaded:
                    print("成功从轨迹文件加载缓存的环境数据")
                    current_structure = last_atoms
                else:
                    print("无法从轨迹文件加载缓存的环境数据")
                
                # 获取日志文件的最后一步
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
                print(f"恢复失败: {e}")
                start_step = 0
        else:
            start_step = 0



        # 构建邻居表（如果未加载）
        if not cached_loaded or self.neighbor_table is None:
            self.neighbor_table = self._build_neighbor_table(current_structure)
        
        # 初始化缓存特征（如果未加载）
        if not cached_loaded or self.cached_features is None:
            self._initialize_cached_features(current_structure)
            
        self._precompute_feature_update_indices(current_structure)
        
        #current_energy = self.predict_energy(current_structure)

        # 恢复时使用轨迹文件中保存的能量，避免重新计算
        if resume and cached_loaded:
            if 'energy' not in current_structure.info:
                print("警告: 轨迹文件中的结构缺少能量信息，重新计算")
                current_energy = self.predict_energy(current_structure)
            else:
                current_energy = current_structure.info['energy']
            
            # 如果加载了原子能量，确保总能量一致
            if hasattr(self, 'atomic_energies') and self.atomic_energies is not None:
                self.total_energy = self.atomic_energies.sum().item()
                if abs(self.total_energy - current_energy) > 1e-4:
                    print(f"警告: 加载的原子能量({self.total_energy:.5f})与结构能量({current_energy:.5f})不一致，使用结构能量")
                    current_energy = self.total_energy
        else:
            current_energy = self.predict_energy(current_structure)

        k_B = 8.617e-5
        T_eV = k_B * self.temperature

        # 写入模式：追加
        write_mode = 'a' if resume else 'w'

        # 清空日志文件（仅在 resume=False 时）
        if not resume:
            with open(log_file, 'w') as f:
                pass  # 清空日志文件
            

        self._initialize_element_indices(current_structure)
        
        
        # 开始轨迹和日志写入
        with Trajectory(trajectory_file, mode=write_mode) as traj:
            # 如果不是 resume，写入初始结构
            if not resume:
                current_structure.info['energy'] = current_energy
                # 保存环境张量到结构
                self._save_cached_features_to_atoms(current_structure)
                traj.write(current_structure)

            # 进行 MC 循环
            results = []
            progression_name = "Monte Carlo Progress of " + str(self.temperature) + "K"
            for step in tqdm(range(start_step, start_step + self.max_steps), desc=progression_name, file=sys.stdout):

                if random.random() < self.swap_ratio:
                    new_structure, modified_indices, old_symbols, new_symbols = self.random_swap(current_structure)
                else:
                    new_structure, modified_indices, old_symbol, new_symbol = self.random_replace(current_structure)
                    old_symbols = [old_symbol]
                    new_symbols = [new_symbol]
                

                # 备份缓存
                cached_features_backup = self.cached_features.clone()
                atomic_energies_backup = self.atomic_energies.clone()
                current_energy = self.total_energy

                # 增量更新缓存
                
                self._update_cached_features(modified_indices, old_symbols, new_symbols)

                # 计算新能量
                new_energy = self._update_energy(modified_indices)
                delta_energy = new_energy - current_energy

                # Metropolis 判断
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

                        # 从旧元素列表中移除
                        if idx in self.element_indices[old_sym]:
                            self.element_indices[old_sym].remove(idx)

                        # 添加到新元素列表
                        self.element_indices[new_sym].append(idx)
                        

                        # 更新索引到元素的映射
                        self.index_to_element[idx] = new_sym
                    current_structure = new_structure.copy()

                    current_energy = new_energy           
                    

                    



                else:
                    # 恢复缓存
                    self.cached_features = cached_features_backup
                    self.atomic_energies = atomic_energies_backup
                    self.total_energy = current_energy

                if step == 1000 and val1000 == True:  # 注意：第100步是索引99
                    # 重新生成环境张量
                    df = self.analyzer.create_atomic_environment_df(current_structure,PBC_layers=1)
                    current_features = torch.tensor(df.values, dtype=torch.float32).to(self.device)

                    # 检查列顺序是否一致
                    if set(df.columns) != set(self.column_map.keys()):
                        raise ValueError("Column mismatch between cached and recalculated features.")

                    # 比较张量
                    if not torch.allclose(self.cached_features, current_features, atol=1e-8):
                        raise ValueError("Cached features do not match recalculated features at step 1000.")     

                if (step+1) % save_interval == 0 and step != start_step:
                    current_structure.info['energy'] = current_energy
                    #self._save_cached_features_to_atoms(current_structure)
                    #print(step,self.max_steps,start_step)
                    if step + 1 == self.max_steps + start_step - 1:
                        self._save_cached_features_to_atoms(current_structure)
                        #print(current_structure.arrays)
                    traj.write(current_structure)

        return results

    def _save_cached_features_to_atoms(self, atoms: Atoms):
        """将环境数据高效保存到 atoms.info 字典中"""
        # 清除旧数据
        keys_to_remove = [
            'cached_features', 'column_names', 'neighbor_table', 
            'atomic_energies', 'cached_features_compressed'
        ]
        for key in keys_to_remove:
            if key in atoms.info:
                del atoms.info[key]

        # 保存环境张量
        if self.cached_features is not None:
            # 转换为 numpy 数组
            cached_features_np = self.cached_features.cpu().numpy()

            # 压缩数据
            compressed = zlib.compress(pickle.dumps(cached_features_np))
            b64_encoded = base64.b64encode(compressed).decode('ascii')

            # 存储到 info
            atoms.info['cached_features_compressed'] = b64_encoded
            atoms.info['cached_features_shape'] = cached_features_np.shape
            atoms.info['cached_features_dtype'] = str(cached_features_np.dtype)

        # 保存列名
        if self.column_map is not None:
            column_names = list(self.column_map.keys())
            atoms.info['column_names'] = column_names

        # 保存邻居表 - 使用高效的序列化格式
        if self.neighbor_table is not None:
            # 转换为可序列化格式
            serializable_table = {}
            for i, layers in self.neighbor_table.items():
                serializable_table[str(i)] = [list(layer) for layer in layers]
            atoms.info['neighbor_table'] = serializable_table

        # 保存原子能量
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
        """实时保存单步模拟结果到日志文件（覆盖文件时写入表头，后续追加）"""
        with open(output_log, 'a') as f:  # 使用追加模式
            if f.tell() == 0:  # 如果文件为空（首次写入）
                f.write("Step\tAttempted_E\tCurrent_E\tDelta_E\tP_accept\tAccepted\n")
            f.write(f"{step_info['step']}\t"
                    f"{step_info['attempted_energy']:.6f}\t"
                    f"{step_info['current_energy']:.6f}\t"
                    f"{step_info['delta_energy']:.6f}\t"
                    f"{step_info['metropolis_prob']:.6f}\t"
                    f"{int(step_info['accepted'])}\n")




def read_enhanced_log_file(log_file: str) -> Dict[str, np.ndarray]:
    """读取增强版日志文件（包含接受状态）"""
    data = defaultdict(list)
    with open(log_file, 'r') as f:
        headers = next(f).strip().split('\t')  # 读取标题行
        for line in f:
            values = line.strip().split('\t')
            for header, value in zip(headers, values):
                if header == 'Step':
                    data[header].append(int(value))
                elif header == 'Accepted':
                    data[header].append(bool(int(value)))
                else:
                    data[header].append(float(value))
    
    # 转换为numpy数组
    return {k: np.array(v) for k, v in data.items()}

def calculate_running_acceptance(steps: np.ndarray, accepted: np.ndarray, 
                                window_size: int = 100) -> np.ndarray:
    """计算滑动窗口接受率"""
    running_acceptance = np.zeros_like(steps, dtype=float)
    for i in range(len(steps)):
        start = max(0, i - window_size + 1)
        running_acceptance[i] = np.mean(accepted[start:i+1])
    return running_acceptance

def plot_energy(log_data):
    # 计算动态接受率
    window_size = min(500, len(log_data['Step'])//10)  # 自适应窗口大小
    running_acceptance = calculate_running_acceptance(
        log_data['Step'], 
        log_data['Accepted'],
        window_size=window_size
    )

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制能量变化 (上方子图)
    color_energy = 'tab:blue'
    ax1.set_ylabel('Energy (eV)', color=color_energy)
    ax1.plot(log_data['Step'], log_data['Current_E'], 
            label='Energy', color=color_energy, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_energy)

    # 绘制接受率 (下方子图)
    color_acceptance = 'tab:purple'
    ax2.set_xlabel('MC Step')
    ax2.set_ylabel('Acceptance Rate', color=color_acceptance)
    ax2.plot(log_data['Step'], running_acceptance, 
            color=color_acceptance, label='Acceptance Rate')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color_acceptance)
    ax2.set_ylim(0, 1)

    # 添加标题和图例
    plt.suptitle('Monte Carlo Simulation: Energy Evolution and Acceptance Rate')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')

    # 调整布局
    plt.tight_layout()
    plt.show()