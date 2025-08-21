import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase import Atoms
from itertools import combinations_with_replacement
import numpy as np
import pandas as pd
from ase.neighborlist import NeighborList
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import re
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
import os
import glob
    
def plot_csro(df, target_marker_count=50, window_size=500):
    """
    Plot CSRO parameters for all element pairs as line charts
    - Same element pairs: Solid line with consistent color, no markers
    - Different element pairs: Solid line (first element's color), markers (second element's color)
    - Total markers ≈ target_marker_count
    - Legend shows both line and marker styles
    """
    plt.figure(figsize=(14, 8))

    # Automatically parse element pairs and assign colors
    color_map = {}  # Element → color
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Matplotlib default colors
    current_color_index = 0

    # Collect legend items
    handles = []
    labels = []

    for col in df.columns:
        if col.startswith("α_"):
            # Parse element pair (e.g., α_TiTi → Ti and Ti)
            elements_part = col.split('_')[1]  # Split by underscore
            elements = re.findall(r'([A-Z][a-z]*)', elements_part)  # Extract element symbols
            if len(elements) < 2:
                print(f"Column name format error: {col}")
                continue
            element1, element2 = elements[0], elements[1]

            # Handle NaN and apply moving average
            smoothed = df[col].fillna(method='ffill').rolling(window=window_size, center=True).mean()

            # Get element color
            def get_element_color(element):
                nonlocal current_color_index
                if element not in color_map:
                    color_map[element] = default_colors[current_color_index % len(default_colors)]
                    current_color_index += 1
                return color_map[element]

            # Line color: First element's color
            line_color = get_element_color(element1)
            # Marker color: Second element's color
            marker_color = get_element_color(element2)

            # Process same element pairs
            if element1 == element2:
                # Same element pair: Solid line, no markers
                line, = plt.plot(smoothed, color=line_color, label=col, linewidth=2)
                handles.append(line)
                labels.append(col)
            else:
                # Different element pair: Line (first element color), markers (second element color)
                line, = plt.plot(smoothed, color=line_color, linewidth=2)
                # Dynamically calculate marker spacing (≈ target_marker_count markers)
                total_points = len(smoothed)
                if total_points < target_marker_count:
                    indices = np.arange(total_points)  # Use all points if total < target
                else:
                    step = total_points // target_marker_count  # Calculate step size
                    indices = np.arange(0, total_points, step)  # Select every step-th point

                # Add markers
                x_markers = indices
                y_markers = smoothed.iloc[x_markers]
                plt.scatter(x_markers, y_markers, color=marker_color, s=5, zorder=5)

                # Create custom legend entry (line + marker)
                custom_line = Line2D([0], [0], color=line_color, linewidth=2, marker='o', 
                                    markerfacecolor=marker_color, markersize=6)
                handles.append(custom_line)
                labels.append(col)

    # Chart formatting
    plt.title("CSRO Parameters Over Frames", fontsize=14)
    plt.xlabel("Frame Index", fontsize=12)
    plt.ylabel("Warren-Cowley Parameter (α)", fontsize=12)
    plt.legend(handles, labels, loc="upper right", fontsize=10, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('csro.png', dpi=300)
    plt.show()

def plot_csro_by_temperature(df):
    """
    Plot CSRO parameters for temperature-varying trajectory (one data point per temperature)
    """
    plt.figure(figsize=(14, 8))

    # Automatically parse element pairs and assign colors
    color_map = {}  # Element → color
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Matplotlib default colors
    current_color_index = 0

    # Collect legend items
    handles = []
    labels = []
    
    # Get temperature list (index)
    temperatures = df.index.values

    for col in df.columns:
        if col.startswith("α_"):
            # Parse element pair (e.g., α_TiTi → Ti and Ti)
            elements_part = col.split('_')[1]  # Split by underscore
            elements = re.findall(r'([A-Z][a-z]*)', elements_part)  # Extract element symbols
            if len(elements) < 2:
                print(f"Column name format error: {col}")
                continue
            element1, element2 = elements[0], elements[1]

            # Get CSRO values for this element pair
            values = df[col].values

            # Get element color
            def get_element_color(element):
                nonlocal current_color_index
                if element not in color_map:
                    color_map[element] = default_colors[current_color_index % len(default_colors)]
                    current_color_index += 1
                return color_map[element]

            # Line color: First element's color
            line_color = get_element_color(element1)
            # Marker color: Second element's color
            marker_color = get_element_color(element2)

            # Process same element pairs
            if element1 == element2:
                # Same element pair: Solid line, no markers
                line, = plt.plot(temperatures, values, color=line_color, label=col, linewidth=2)
                handles.append(line)
                labels.append(col)
            else:
                # Different element pair: Line (first element color), markers (second element color)
                line, = plt.plot(temperatures, values, color=line_color, linewidth=2)
                # Add marker for every temperature point
                plt.scatter(temperatures, values, color=marker_color, s=30, zorder=5)

                # Create custom legend entry (line + marker)
                custom_line = Line2D([0], [0], color=line_color, linewidth=2, marker='o', 
                                    markerfacecolor=marker_color, markersize=8)
                handles.append(custom_line)
                labels.append(col)

    # Chart formatting
    plt.title("CSRO Parameters by Temperature", fontsize=14)
    plt.xlabel("Temperature (K)", fontsize=12)
    plt.ylabel("Warren-Cowley Parameter (α)", fontsize=12)
    plt.legend(handles, labels, loc="upper right", fontsize=10, ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('csro_by_T', dpi=300)
    plt.show()

def count_element_pairs(df, layers=1):
    """
    Count element pair coordination numbers within specified neighbor layers
    """
    pair_dict = defaultdict(int)
    group_pairs = [(0, i) for i in range(1, layers + 1)]
    
    for g1, g2 in group_pairs:
        cols_g1 = [col for col in df.columns if col.startswith(f"{g1}_")]
        cols_g2 = [col for col in df.columns if col.startswith(f"{g2}_")]
        
        for _, row in df.iterrows():
            for col_g1 in cols_g1:
                val_g1 = row[col_g1]
                element_g1 = col_g1.split('_')[1]
                for col_g2 in cols_g2:
                    val_g2 = row[col_g2]
                    element_g2 = col_g2.split('_')[1]
                    key = f"{element_g1}_{element_g2}"
                    pair_dict[key] += val_g1 * val_g2  # Actual pair count
    # Normalization
    total = sum(pair_dict.values())
    if total > 0:
        normalized_dict = {k: v / total for k, v in pair_dict.items()}
    else:
        normalized_dict = {}
    data = dict(normalized_dict)
    unique_data = {}

    for key in data:
        elements = key.split('_')
        sorted_elements = sorted(elements)
        new_key = '_'.join(sorted_elements)
        if new_key not in unique_data:
            unique_data[new_key] = data[key]
    return unique_data

def calculate_concentrations(df):
    """
    Calculate element concentrations in group 0 (dimensionless)
    """
    cols_g0 = [col for col in df.columns if col.startswith("0_")]
    elements = set(col.split('_')[1] for col in cols_g0)
    total_per_element = {e: df[f"0_{e}"].sum() for e in elements}
    total_group = sum(df[col].sum() for col in cols_g0)
    return {e: count / total_group for e, count in total_per_element.items()}

def CSRO(dict1, dict2):
    """
    Calculate Warren-Cowley parameters from:
    dict1: Normalized pair counts (from count_element_pairs)
    dict2: Element concentrations (from calculate_concentrations)
    """
    new_dict = {}

    for key, v3 in dict1.items():
        # Split key into two elements
        elem1, elem2 = key.split('_')
        # Get corresponding element concentrations
        v1 = dict2[elem1]
        v2 = dict2[elem2]
        # Calculate new value (α = 1 - P_ij/(c_i·c_j))
        new_value = 1 - v3 / (v1 * v2)
        new_dict[key] = new_value

    return new_dict

def calculate_csro(atom, analyzer, layers=1):
    """
    Calculate CSRO parameters for a single atomic configuration
    """
    df = analyzer.create_atomic_environment_df(atom, PBC_layers=1)
    pair_counts = count_element_pairs(df, layers=layers)
    concentrations = calculate_concentrations(df)
    return CSRO(pair_counts, concentrations)

def generate_csro_table(traj_file, analyzer, n_neighbor=1):
    """
    Generate CSRO parameter table for a trajectory file (multi-frame)
    """
    with Trajectory(traj_file, mode='r') as traj:
        n_frames = len(traj)
        first_frame = traj[0]
        symbols = first_frame.get_chemical_symbols()
        unique_elements = sorted(set(symbols))
        element_pairs = list(combinations_with_replacement(unique_elements, 2))  # Unique pairs

        args_list = [(traj_file, idx, n_neighbor, analyzer) for idx in range(n_frames)]

        results = []
        with ProcessPoolExecutor() as executor:
            for _, wc_params in tqdm(
                executor.map(process_frame, args_list),
                total=n_frames,
                desc="CSRO Analysis Progress"
            ):
                frame_data = {}
                for pair in element_pairs:
                    key = f"{pair[0]}_{pair[1]}"
                    frame_data[f"α_{pair[0]}{pair[1]}"] = wc_params.get(key, None)
                results.append(frame_data)

    return pd.DataFrame(results)

def process_frame(args):
    """Process a single frame (worker function for parallel processing)"""
    traj_file, frame_idx, n_neighbor, analyzer = args
    with Trajectory(traj_file, 'r') as traj:
        atoms = traj[frame_idx].copy()
    return frame_idx, calculate_csro(atoms, analyzer, layers=n_neighbor)

def calculate_temperature_csro(traj_file, analyzer, n_neighbor=1, skip_ratio=0.3):
    """
    Calculate average CSRO parameters for a single temperature trajectory
    :param traj_file: Trajectory file path
    :param analyzer: StructureAnalyzer instance
    :param n_neighbor: Neighbor layers to consider
    :param skip_ratio: Fraction of initial frames to skip (for equilibration)
    :return: Dictionary of average CSRO values for element pairs
    """
    with Trajectory(traj_file, mode='r') as traj:
        n_frames = len(traj)
        # Calculate frames to skip
        skip_frames = int(n_frames * skip_ratio)
        # Only process frames after equilibration
        frame_indices = range(skip_frames, n_frames)
        
        first_frame = traj[0]
        symbols = first_frame.get_chemical_symbols()
        unique_elements = sorted(set(symbols))
        element_pairs = list(combinations_with_replacement(unique_elements, 2))

        args_list = [(traj_file, idx, n_neighbor, analyzer) for idx in frame_indices]

        results = []
        with ProcessPoolExecutor() as executor:
            for _, wc_params in tqdm(
                executor.map(process_frame, args_list),
                total=len(frame_indices),
                desc=f"Processing {os.path.basename(traj_file)}"
            ):
                frame_data = {}
                for pair in element_pairs:
                    key = f"{pair[0]}_{pair[1]}"
                    frame_data[f"α_{pair[0]}{pair[1]}"] = wc_params.get(key, None)
                results.append(frame_data)
        
        # Calculate average CSRO parameters
        df_temp = pd.DataFrame(results)
        return df_temp.mean().to_dict()

def generate_csro_table_by_temperature(base_dir, traj_keyword, analyzer, n_neighbor=1, skip_ratio=0.3):
    """
    Generate CSRO parameter table across multiple temperatures
    :param base_dir: Root directory containing temperature subfolders (T_*)
    :param traj_keyword: Base name pattern for trajectory files
    :param analyzer: StructureAnalyzer instance
    :param n_neighbor: Neighbor layers to consider
    :param skip_ratio: Fraction of initial frames to skip (for equilibration)
    :return: DataFrame with temperature index and CSRO parameter columns
    """
    # Find all temperature directories
    temp_dirs = glob.glob(os.path.join(base_dir, "T_*"))
    temp_dirs = sorted(temp_dirs, 
                          key=lambda s: int(re.search(r'T_(\d+)', s).group(1)))
    temperature_data = []
    
    for temp_dir in temp_dirs:
        # Extract temperature from directory name (e.g., "T_3000K" → 3000.0)
        dir_name = os.path.basename(temp_dir)
        temp_str = dir_name.split('_')[1]  # Format like "3000K"
        try:
            temperature = float(temp_str[:-1])  # Remove 'K' suffix
        except:
            print(f"Skipping invalid temperature directory: {dir_name}")
            continue
        
        # Build trajectory file path pattern
        traj_pattern = os.path.join(temp_dir, f"{traj_keyword}_{temp_str}.traj")
        traj_files = glob.glob(traj_pattern)
        if not traj_files:
            print(f"No matching trajectory in {dir_name}: {traj_pattern}")
            continue
        traj_file = traj_files[0]  # Use first match
        
        # Calculate average CSRO for this temperature
        avg_csro = calculate_temperature_csro(traj_file, analyzer, n_neighbor, skip_ratio)
        if avg_csro:
            avg_csro['Temperature'] = temperature
            temperature_data.append(avg_csro)
    
    # Create and sort DataFrame
    df = pd.DataFrame(temperature_data)
    df.set_index('Temperature', inplace=True)
    df.sort_index(inplace=True)
    return df