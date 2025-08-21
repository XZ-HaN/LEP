import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
import os
import glob

def calculate_temperature_energy(traj_file, skip_ratio=0.3):
    """
    Calculate average energy for a single temperature trajectory file
    :param traj_file: Path to trajectory file
    :param skip_ratio: Ratio of initial frames to skip (for equilibration)
    :return: Average energy per atom at this temperature (in eV/atom)
    """
    with Trajectory(traj_file, mode='r') as traj:
        n_frames = len(traj)
        if n_frames == 0:
            return None
        
        # Calculate number of frames to skip
        skip_frames = int(n_frames * skip_ratio)
        # Only calculate for the last (1 - skip_ratio) portion of frames
        energies = []
        for i in range(skip_frames, n_frames):
            atoms = traj[i]
            energy = atoms.info['energy'] / len(atoms)  # Per-atom energy
            energies.append(energy)
        
        # Calculate average energy
        if energies:
            return np.mean(energies)
        return None

def generate_energy_table(base_dir, traj_keyword, skip_ratio=0.3):
    """
    Generate energy table varying with temperature
    :param base_dir: Root directory of trajectory folders (contains T_* subdirectories)
    :param traj_keyword: Keyword for trajectory filenames (e.g., "alloy_annealing")
    :param skip_ratio: Ratio of initial frames to skip
    :return: DataFrame containing temperatures and average energies
    """
    # Get all temperature subdirectories
    temp_dirs = glob.glob(os.path.join(base_dir, "T_*"))
    temperature_data = []
    
    for temp_dir in temp_dirs:
        # Extract temperature from directory name (e.g., "T_3000K" -> 3000)
        dir_name = os.path.basename(temp_dir)
        temp_str = dir_name.split('_')[1]  # Format like "3000K"
        try:
            # Remove 'K' suffix and convert to float
            temperature = float(temp_str[:-1])
        except:
            print(f"Cannot extract temperature from directory name {dir_name}, skipping")
            continue
        
        # Build trajectory file path: e.g., "alloy_annealing_3000K.traj"
        traj_pattern = os.path.join(temp_dir, f"{traj_keyword}_{temp_str}.traj")
        traj_files = glob.glob(traj_pattern)
        if not traj_files:
            print(f"No matching trajectory files found in {temp_dir}: {traj_pattern}")
            continue
        traj_file = traj_files[0]  # Take first matching file
        
        # Calculate average energy at this temperature
        avg_energy = calculate_temperature_energy(traj_file, skip_ratio)
        if avg_energy is not None:
            temperature_data.append({
                'Temperature': temperature,
                'Energy': avg_energy
            })
    
    # Create DataFrame and sort by temperature
    df = pd.DataFrame(temperature_data)
    df.sort_values('Temperature', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_heat_capacity(df):
    """
    Calculate heat capacity (dE/dT)
    Using central difference method: dE/dT ≈ (E_{i+1} - E_{i-1}) / (T_{i+1} - T_{i-1})
    :param df: DataFrame containing temperatures and energies
    :return: DataFrame with heat capacity column added
    """
    # Ensure sorted by temperature
    df = df.sort_values('Temperature').reset_index(drop=True)
    
    # Initialize heat capacity column
    df['Heat_Capacity'] = np.nan
    
    # Calculate derivative for middle points using central difference
    n = len(df)
    for i in range(1, n-1):
        dE = df.at[i+1, 'Energy'] - df.at[i-1, 'Energy']
        dT = df.at[i+1, 'Temperature'] - df.at[i-1, 'Temperature']
        if dT > 0:  # Avoid division by zero
            df.at[i, 'Heat_Capacity'] = dE / dT
    
    # For endpoints, use forward/backward difference
    if n > 1:
        # First point: forward difference
        dE = df.at[1, 'Energy'] - df.at[0, 'Energy']
        dT = df.at[1, 'Temperature'] - df.at[0, 'Temperature']
        if dT > 0:
            df.at[0, 'Heat_Capacity'] = dE / dT
        
        # Last point: backward difference
        dE = df.at[n-1, 'Energy'] - df.at[n-2, 'Energy']
        dT = df.at[n-1, 'Temperature'] - df.at[n-2, 'Temperature']
        if dT > 0:
            df.at[n-1, 'Heat_Capacity'] = dE / dT
    
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

def plot_heat_capacity(df):
    """Plot heat capacity vs. temperature curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Temperature'], df['Heat_Capacity'], 'ro-', linewidth=2, markersize=8)
    
    plt.title('Heat Capacity vs. Temperature', fontsize=14)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('dE/dT (eV/K/atom)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add peak information
    if not df['Heat_Capacity'].isnull().all():
        max_capacity = df['Heat_Capacity'].max()
        max_temp = df.loc[df['Heat_Capacity'].idxmax(), 'Temperature']
        plt.text(0.05, 0.95, f'Max dE/dT: {max_capacity:.3g} eV/(K·atom) at {max_temp:.0f}K',
                 transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('heat_by_T.png', dpi=300)
    plt.show()

def main():
    """Main function: execute full analysis workflow"""
    # Configuration parameters
    base_dir = "annealing_results"  # Root directory of trajectory folders
    traj_keyword = "alloy_annealing"  # Trajectory file keyword
    skip_ratio = 0.3  # Skip first 30% of frames as equilibration
    
    # Generate energy table
    df_energy = generate_energy_table(base_dir, traj_keyword, skip_ratio)
    
    if df_energy.empty:
        print("No valid data found, please check paths and files")
        return
    
    # Save data
    df_energy.to_csv("energy_vs_temperature.csv", index=False)
    
    # Plot energy vs temperature
    plot_energy_temperature(df_energy)
    
    # Calculate and plot heat capacity
    df_heat = calculate_heat_capacity(df_energy)
    plot_heat_capacity(df_heat)
    
    # Save heat capacity data
    df_heat.to_csv("heat_capacity.csv", index=False)

if __name__ == "__main__":
    main()