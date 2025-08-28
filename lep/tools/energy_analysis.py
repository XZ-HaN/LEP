import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
import os
import glob

# Boltzmann constant in eV/K
k_B = 8.617333262145e-5

def calculate_temperature_energy(log_file, traj_file, skip_ratio=0.3):
    """
    Calculate average energy and energy variance from log file
    :param log_file: Path to log file (CSV format)
    :param traj_file: Path to trajectory file (to get atom count)
    :param skip_ratio: Ratio of initial frames to skip
    :return: (avg_energy_per_atom, variance, n_atoms)
    """
    # Get atom count from trajectory file
    try:
        with Trajectory(traj_file, mode='r') as traj:
            if len(traj) > 0:
                n_atoms = len(traj[0])
            else:
                print(f"Empty trajectory file: {traj_file}")
                return None, None, None
    except Exception as e:
        print(f"Error reading trajectory file {traj_file}: {e}")
        return None, None, None
    
    # Read log file and process all MC steps
    try:
        df_log = pd.read_csv(log_file, sep='\t')  # 使用制表符分隔
        
        # Check required columns
        required_columns = ['Step', 'Current_E']
        if not all(col in df_log.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df_log.columns]
            print(f"Missing columns in {log_file}: {', '.join(missing)}")
            return None, None, None
        
        # Skip initial equilibration period
        n_frames = len(df_log)
        skip_frames = int(n_frames * skip_ratio)
        if skip_frames >= n_frames:
            print(f"Too many frames skipped ({skip_frames}/{n_frames}) in {log_file}")
            return None, None, None
        
        # Get all energy values after equilibration period
        total_energies = df_log['Current_E'].iloc[skip_frames:]
        
        if len(total_energies) == 0:
            print(f"No energy data found after skipping initial frames in {log_file}")
            return None, None, None
        
        # Calculate average energy per atom
        avg_energy_total = total_energies.mean()
        avg_energy_per_atom = avg_energy_total / n_atoms
        
        # Calculate variance for fluctuation formula
        U_mean = total_energies.mean()
        U_sq_mean = (total_energies ** 2).mean()
        variance = U_sq_mean - U_mean ** 2
        
        return avg_energy_per_atom, variance, n_atoms
        
    except Exception as e:
        print(f"Error processing log file {log_file}: {e}")
        return None, None, None

def generate_energy_table(base_dir, log_keyword, skip_ratio=0.3):
    """
    Generate energy table using log files, sorted by temperature
    :param base_dir: Root directory containing T_* subdirectories
    :param log_keyword: Base filename for log files (e.g., "alloy_annealing")
    :param skip_ratio: Ratio of initial frames to skip
    :return: DataFrame containing temperatures, average energies, variances and atom counts
    """
    temp_dirs = glob.glob(os.path.join(base_dir, "T_*"))
    
    # 创建包含温度信息的列表，用于排序
    temp_info = []
    for temp_dir in temp_dirs:
        dir_name = os.path.basename(temp_dir)
        temp_str = dir_name.split('_')[1]
        try:
            temperature = float(temp_str[:-1])  # Remove 'K' suffix
            temp_info.append((temperature, temp_dir))
        except:
            print(f"Cannot parse temperature from {dir_name}, skipping")
            continue
    
    # 按照温度升序排序
    temp_info.sort(key=lambda x: x[0])
    
    temperature_data = []
    
    for temperature, temp_dir in temp_info:
        temp_str = f"{int(temperature)}K"  # 重建温度字符串
        
        # Find log file and corresponding trajectory file
        log_pattern = os.path.join(temp_dir, f"{log_keyword}_{temp_str}.txt")
        log_files = glob.glob(log_pattern)
        if not log_files:
            print(f"No log file found: {log_pattern}")
            continue
        log_file = log_files[0]
        
        # Find corresponding trajectory file for atom count
        traj_pattern = os.path.join(temp_dir, f"{log_keyword}_{temp_str}.traj")
        traj_files = glob.glob(traj_pattern)
        if not traj_files:
            print(f"No trajectory file found for atom count: {traj_pattern}")
            continue
        traj_file = traj_files[0]
        
        # Calculate average energy and variance
        avg_energy, variance, n_atoms = calculate_temperature_energy(log_file, traj_file, skip_ratio)
        if avg_energy is not None and variance is not None:
            temperature_data.append({
                'Temperature': temperature,
                'Energy': avg_energy,
                'Variance': variance,
                'N_atoms': n_atoms
            })
            print(f"Processed T={temperature}K: avg energy = {avg_energy:.6f} eV/atom")
        else:
            print(f"Failed to process temperature {temperature}K")
    
    # Create DataFrame (already sorted by temperature)
    if not temperature_data:
        print("No valid temperature data found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(temperature_data)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_heat_capacity_fluctuation(df):
    """
    Calculate heat capacity using fluctuation formula
    C_v = [<U²> - <U>²] / (k_B * T² * N)
    :param df: DataFrame containing temperatures, variances, and atom counts
    :return: DataFrame with heat capacity column added
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Calculate heat capacity per atom
    df['Heat_Capacity_Fluctuation'] = df.apply(
        lambda row: row['Variance'] / (k_B * row['Temperature']**2 * row['N_atoms']),
        axis=1
    )
    
    return df

def calculate_heat_capacity_derivative(df):
    """
    Calculate heat capacity using derivative method
    :param df: DataFrame containing temperatures and energies
    :return: DataFrame with heat capacity column added
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Initialize heat capacity column
    df['Heat_Capacity_Derivative'] = np.nan
    
    # Calculate derivative for middle points using central difference
    n = len(df)
    for i in range(1, n-1):
        dE = df.at[i+1, 'Energy'] - df.at[i-1, 'Energy']
        dT = df.at[i+1, 'Temperature'] - df.at[i-1, 'Temperature']
        if dT > 0:  # Avoid division by zero
            df.at[i, 'Heat_Capacity_Derivative'] = dE / dT
    
    # For endpoints, use forward/backward difference
    if n > 1:
        # First point: forward difference
        dE = df.at[1, 'Energy'] - df.at[0, 'Energy']
        dT = df.at[1, 'Temperature'] - df.at[0, 'Temperature']
        if dT > 0:
            df.at[0, 'Heat_Capacity_Derivative'] = dE / dT
        
        # Last point: backward difference
        dE = df.at[n-1, 'Energy'] - df.at[n-2, 'Energy']
        dT = df.at[n-1, 'Temperature'] - df.at[n-2, 'Temperature']
        if dT > 0:
            df.at[n-1, 'Heat_Capacity_Derivative'] = dE / dT
    
    return df

def plot_energy_temperature(df):
    """Plot energy vs. temperature curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], df['Energy'], 'bo-', linewidth=2, markersize=8)
    
    plt.title('Average Energy vs. Temperature', fontsize=14)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Energy per Atom (eV)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text annotation
    min_energy = df['Energy'].min()
    max_energy = df['Energy'].max()
    plt.text(0.05, 0.95, f'Energy Range: {min_energy:.4f} ~ {max_energy:.4f} eV/atom',
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('energy_by_T.png', dpi=300)
    plt.show()

def plot_heat_capacity(df, method='both'):
    """
    Plot heat capacity vs. temperature curve(s)
    :param method: 'derivative', 'fluctuation', or 'both'
    """
    plt.figure(figsize=(12, 8))
    
    title_added = False
    
    # Plot fluctuation method if selected
    if method in ['fluctuation', 'both'] and 'Heat_Capacity_Fluctuation' in df.columns:
        plt.plot(df['Temperature'], df['Heat_Capacity_Fluctuation'], 
                 'ro-', linewidth=2, markersize=8, label='Fluctuation Method')
        
        # Add peak info for fluctuation method
        if not df['Heat_Capacity_Fluctuation'].isnull().all():
            max_cap = df['Heat_Capacity_Fluctuation'].max()
            max_temp = df.loc[df['Heat_Capacity_Fluctuation'].idxmax(), 'Temperature']
            plt.text(0.05, 0.90, f'Fluctuation Max: {max_cap:.3g} eV/(K·atom) at {max_temp:.0f}K',
                     transform=plt.gca().transAxes, verticalalignment='top')
            title_added = True
    
    # Plot derivative method if selected
    if method in ['derivative', 'both'] and 'Heat_Capacity_Derivative' in df.columns:
        plt.plot(df['Temperature'], df['Heat_Capacity_Derivative'], 
                 'gx--', linewidth=1.5, markersize=6, label='Derivative Method')
        
        # Add peak info for derivative method
        if not df['Heat_Capacity_Derivative'].isnull().all():
            max_cap = df['Heat_Capacity_Derivative'].max()
            max_temp = df.loc[df['Heat_Capacity_Derivative'].idxmax(), 'Temperature']
            plt.text(0.05, 0.85 if title_added else 0.90, 
                     f'Derivative Max: {max_cap:.3g} eV/(K·atom) at {max_temp:.0f}K',
                     transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.title('Heat Capacity vs. Temperature', fontsize=14)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Heat Capacity (eV/K/atom)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if method == 'both' and ('Heat_Capacity_Fluctuation' in df.columns or 
                             'Heat_Capacity_Derivative' in df.columns):
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('heat_capacity.png', dpi=300)
    plt.show()

def main():
    """Main function: execute full analysis workflow"""
    # Configuration parameters (set as variables in main)
    base_dir = "annealing_results"  # Root directory of trajectory folders
    log_keyword = "alloy_annealing"  # Base filename for log files
    skip_ratio = 0.5  # Skip first 30% of frames for equilibration
    method = 'derivative'    # Heat capacity calculation method: 'derivative', 'fluctuation', or 'both'
    
    print("Starting analysis of MC simulation data...")
    print(f"Base directory: {base_dir}")
    print(f"Log file keyword: {log_keyword}")
    print(f"Skipping first {skip_ratio*100:.0f}% of frames for equilibration")
    print(f"Heat capacity calculation method: {method}")
    
    # Generate energy table (already sorted by temperature)
    df_energy = generate_energy_table(base_dir, log_keyword, skip_ratio)
    
    if df_energy.empty:
        print("No valid data found, please check paths and files")
        return
    
    # Save energy data
    energy_file = "energy_vs_temperature.csv"
    df_energy.to_csv(energy_file, index=False)
    print(f"Saved energy data to: {energy_file}")
    
    # Plot energy vs temperature
    plot_energy_temperature(df_energy)
    
    # Calculate heat capacity using selected methods
    if method in ['fluctuation', 'both']:
        df_heat = calculate_heat_capacity_fluctuation(df_energy)
        # 如果使用两种方法，需要保留df_energy用于导数法
        if method == 'both':
            df_heat = calculate_heat_capacity_derivative(df_heat)
    elif method == 'derivative':
        df_heat = calculate_heat_capacity_derivative(df_energy)
    
    # Plot heat capacity
    plot_heat_capacity(df_heat, method=method)
    
    # Save heat capacity data
    heat_file = "heat_capacity.csv"
    df_heat.to_csv(heat_file, index=False)
    print(f"Saved heat capacity data to: {heat_file}")
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()