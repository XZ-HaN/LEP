import os
import re
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from tqdm import tqdm  # For progress bars

def analyze_trajectories(folder_name, filename_prefix, elements, skip_ratio=0.5):
    """
    Analyze trajectory files and plot average atomic energy contributions vs temperature
    Skips the initial portion of each trajectory (pre-equilibrium) based on skip_ratio
    
    Parameters:
    folder_name: Main folder name
    filename_prefix: Trajectory file prefix
    elements: List of element symbols (e.g., ['C', 'H', 'O'])
    skip_ratio: Fraction of initial frames to skip (0.0-1.0, default=0.5)
    """
    # Validate skip_ratio
    if not (0.0 <= skip_ratio <= 1.0):
        print(f"Warning: Invalid skip_ratio {skip_ratio}. Using default 0.5")
        skip_ratio = 0.5
    
    # Step 1: Collect all temperatures
    print("Scanning for temperature directories...")
    temperatures = set()
    pattern = re.compile(r'T_(\d+)K')
    valid_dirs = []
    
    for subdir in os.listdir(folder_name):
        full_path = os.path.join(folder_name, subdir)
        if os.path.isdir(full_path):
            match = pattern.match(subdir)
            if match:
                temp = int(match.group(1))
                temperatures.add(temp)
                valid_dirs.append(subdir)
    
    if not temperatures:
        print(f"Error: No valid temperature directories found in '{folder_name}'")
        return
    
    temperatures = sorted(temperatures)
    min_temp = min(temperatures)
    max_temp = max(temperatures)
    num_dirs = len(valid_dirs)
    
    print(f"Found {num_dirs} temperature directories")
    print(f"Temperature range: {min_temp}K to {max_temp}K")
    print(f"Temperatures: {temperatures}")
    print(f"Using skip_ratio: {skip_ratio} (skipping first {skip_ratio*100:.0f}% of each trajectory)")
    
    # Step 2: Initialize data storage
    element_energies = {element: {temp: [] for temp in temperatures} for element in elements}
    frame_counts = {temp: [] for temp in temperatures}  # Track frame counts for each trajectory
    
    # Step 3: Process all temperature subdirectories with progress bar
    print("\nProcessing trajectory files:")
    for temp in tqdm(temperatures, desc="Element Energy Analysis"):
        subdir = os.path.join(folder_name, f'T_{temp}K')
        if not os.path.isdir(subdir):
            print(f"Warning: Directory not found - {subdir}")
            continue
            
        # Step 4: Process all trajectory files at current temperature
        for file in os.listdir(subdir):
            if file.startswith(f'{filename_prefix}_{temp}K') and file.endswith('.traj'):
                traj_path = os.path.join(subdir, file)
                try:
                    # Read trajectory file
                    frames = read(traj_path, index=':')
                    total_frames = len(frames)
                    
                    # Calculate frames to skip
                    skip_frames = int(total_frames * skip_ratio)
                    if skip_frames >= total_frames:
                        skip_frames = max(0, total_frames - 1)  # Ensure at least one frame remains
                    
                    # Record frame statistics
                    frame_counts[temp].append((total_frames, skip_frames))
                    
                    # Process only the equilibrium portion (after skip_frames)
                    for atoms in frames[skip_frames:]:
                        symbols = atoms.get_chemical_symbols()
                        atomic_energies = atoms.info.get('atomic_energies', [])
                        
                        # Verify data consistency
                        if len(symbols) != len(atomic_energies):
                            print(f"Warning: Atom-energy mismatch in {file} - {len(symbols)} atoms vs {len(atomic_energies)} energies")
                            continue
                            
                        # Collect element energies
                        for symbol, energy in zip(symbols, atomic_energies):
                            if symbol in element_energies:
                                element_energies[symbol][temp].append(energy)
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    # Print frame skipping statistics
    print("\nFrame skipping statistics:")
    for temp in temperatures:
        if frame_counts[temp]:
            total_original = sum(t for t, s in frame_counts[temp])
            total_skipped = sum(s for t, s in frame_counts[temp])
            total_used = total_original - total_skipped
            print(f"T = {temp}K: {len(frame_counts[temp])} trajectories, "
                  f"Original frames: {total_original}, Used frames: {total_used} "
                  f"({total_used/total_original*100:.1f}%)")
    
    # Step 5: Calculate average energies
    print("\nCalculating average energies...")
    avg_energies = {element: [] for element in elements}
    std_energies = {element: [] for element in elements}  # For error bars
    
    for element in elements:
        for temp in temperatures:
            energies = element_energies[element][temp]
            if energies:
                avg_energies[element].append(np.mean(energies))
                std_energies[element].append(np.std(energies))
            else:
                avg_energies[element].append(np.nan)
                std_energies[element].append(np.nan)
                print(f"Warning: No data for {element} at {temp}K")
    
    # Step 6: Plot results
    print("Generating plot...")
    plt.figure(figsize=(10, 6))

    for element in elements:
        # Filter out temperatures with no data
        valid_temps = []
        valid_energies = []
        valid_stds = []

        for i, temp in enumerate(temperatures):
            if i == len(temperatures)-1: break
            if not np.isnan(avg_energies[element][i]):
                valid_temps.append(temp)
                valid_energies.append(avg_energies[element][i])
                valid_stds.append(std_energies[element][i])

        if valid_temps:
            plt.errorbar(valid_temps, valid_energies,  
                         marker='o', linestyle='-', capsize=5, label=element)

    plt.title(f'Average Atomic Energy Contribution by Element')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Energy (eV)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_name = f'element_energies_skip{int(skip_ratio*100)}.png'
    plot_path = os.path.join(folder_name, plot_name)
    plt.savefig(plot_path, dpi=300)
    print(f"\nResults saved to: {plot_path}")

    # Show plot
    plt.show()


# Example usage
if __name__ == "__main__":
    folder = "2000a_1e6"
    prefix = "alloy_annealing"
    elements = ['Ti', 'Al', 'Nb','Mo','Zr'] 
    skip_fraction = 0.5
    
    analyze_trajectories(folder, prefix, elements)