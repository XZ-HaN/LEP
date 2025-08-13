from .mc import MonteCarloSampler
from ase import Atoms
from typing import Tuple, List, Dict
import os
import shutil
import numpy as np
from tqdm import tqdm
from ase.io import read, Trajectory
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Annealer:
    """
    General-purpose annealing processor for Monte Carlo simulations
    with efficient state transfer between temperatures.
    
    Parameters:
        sampler: Monte Carlo sampler instance (must implement run_monte_carlo method)
        initial_structure: Initial atomic structure (ASE Atoms object)
        start_temp: Starting temperature (K)
        end_temp: Final temperature (K)
        temp_step: Temperature step size (K)
        base_filename: Base name for output files
        output_dir: Output directory (default: 'annealing_results')
        save_interval: Save interval for each MC simulation (default: 100 steps)
        steps_per_temp: Number of MC steps per temperature (default: 1000)
        resume: Resume from previous run (default: False)
        purge_existing: Delete existing output directory (default: False)
    """
    
    def __init__(
        self,
        sampler,
        initial_structure,
        start_temp: float,
        end_temp: float,
        temp_step: float,
        base_filename: str = "anneal",
        output_dir: str = "annealing_results",
        save_interval: int = 100,
        steps_per_temp: int = 1000,
        resume: bool = False,
        purge_existing: bool = False
    ):
        self.sampler = sampler
        self.initial_structure = initial_structure
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.temp_step = temp_step
        self.base_filename = base_filename
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.steps_per_temp = steps_per_temp
        self.resume = resume
        self.purge_existing = purge_existing
        
        # Generate temperature sequence
        self.temperatures = self._generate_temperature_sequence()
        
        # Prepare output directory
        self._prepare_output_directory()
        
        # Initialize annealing progress tracking
        self._initialize_annealing_progress()
        
    def _generate_temperature_sequence(self):
        """Generate temperature sequence (high to low or low to high)"""
        if self.start_temp > self.end_temp:
            # Decreasing temperature sequence
            temps = np.arange(self.start_temp, self.end_temp - 0.1, -self.temp_step)
        else:
            # Increasing temperature sequence
            temps = np.arange(self.start_temp, self.end_temp + 0.1, self.temp_step)
        return temps
    
    def _prepare_output_directory(self):
        """Create output directory structure"""
        if self.purge_existing and os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            logger.info(f"Purged existing output directory: {self.output_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {self.output_dir}")
        
    def _initialize_annealing_progress(self):
        """Initialize annealing progress tracking with resume support"""
        self.start_index = 0
        self.current_structure = self.initial_structure.copy()
        
        if self.resume:
            # Find the last completed temperature
            last_temp, last_file = self._find_last_completed_temperature()
            
            if last_temp is not None:
                # Load the last structure with cached state
                self.current_structure = read(last_file)
                logger.info(f"Resumed from T = {last_temp}K with {len(self.current_structure)} atoms")
                
                # Set starting index to next temperature
                last_index = np.where(self.temperatures == last_temp)[0]
                if len(last_index) > 0:
                    self.start_index = last_index[0] + 1
                else:
                    self.start_index = 0
                
                # Check if annealing is already completed
                if self.start_index >= len(self.temperatures):
                    logger.info("Annealing already completed!")
                    self.temperatures = []
                else:
                    logger.info(f"Continuing from T = {self.temperatures[self.start_index]}K")
            else:
                logger.info("No previous annealing results found, starting from scratch")
                self.resume = False
    
    def _find_last_completed_temperature(self):
        """Find the last completed temperature and its trajectory file"""
        # Traverse temperatures in completion order
        if self.start_temp > self.end_temp:
            # Decreasing temperatures: from high to low
            temp_iter = sorted(self.temperatures, reverse=True)
        else:
            # Increasing temperatures: from low to high
            temp_iter = sorted(self.temperatures)
            
        for temp in temp_iter:
            temp_dir = os.path.join(self.output_dir, f"T_{int(temp)}K")
            traj_file = os.path.join(temp_dir, f"{self.base_filename}_{int(temp)}K.traj")
            
            if os.path.exists(traj_file):
                try:
                    # Verify file contains at least one structure
                    with Trajectory(traj_file, 'r') as traj:
                        if len(traj) > 0:
                            return temp, traj_file
                except Exception as e:
                    logger.warning(f"Error reading {traj_file}: {e}")
                    
        return None, None

    def run_annealing(self):
        """Execute the full annealing process"""
        # Check if temperature sequence is empty
        if len(self.temperatures) == 0:
            logger.warning("No temperatures to process!")
            return {}

        annealing_results = {}
        total_temps = len(self.temperatures)

        # Print annealing parameters summary
        print("\n" + "="*80)
        print(f"Starting Annealing Process")
        print(f"  Start Temperature: {self.start_temp}K")
        print(f"  End Temperature: {self.end_temp}K")
        print(f"  Temperature Step: {self.temp_step}K")
        print(f"  Steps per Temperature: {self.steps_per_temp}")
        print(f"  Total Temperatures: {total_temps}")
        print(f"  Starting from Temperature: {self.temperatures[self.start_index]}K")
        print("="*80 + "\n")

        # Process each temperature
        for i in range(self.start_index, len(self.temperatures)):
            temp = self.temperatures[i]
            current_temp_index = i - self.start_index + 1
            total_current_temps = len(self.temperatures) - self.start_index
            percent_complete = (current_temp_index / total_current_temps) * 100

            # Print current temperature information
            print("\n" + "-"*60)
            print(f"Processing Temperature {current_temp_index}/{total_current_temps} ({percent_complete:.1f}%)")
            print(f"  Current Temperature: {temp}K")
            print(f"  Start Temperature: {self.start_temp}K")
            print(f"  End Temperature: {self.end_temp}K")
            print(f"  Temperature Step: {self.temp_step}K")
            print(f"  Steps per Temperature: {self.steps_per_temp}")
            print("-"*60 + "\n")

            # Create temperature-specific directory
            temp_dir = os.path.join(self.output_dir, f"T_{int(temp)}K")
            os.makedirs(temp_dir, exist_ok=True)

            # Set current temperature and steps
            self.sampler.temperature = temp
            self.sampler.max_steps = self.steps_per_temp

            # Prepare file paths
            traj_file = os.path.join(temp_dir, f"{self.base_filename}_{int(temp)}K.traj")
            log_file = os.path.join(temp_dir, f"{self.base_filename}_{int(temp)}K.txt")

            # Determine resume file
            resume_file = None
            if i > self.start_index:
                # Use previous temperature's cache as resume point
                prev_temp = self.temperatures[i-1]
                prev_cache = os.path.join(
                    os.path.join(self.output_dir, f"T_{int(prev_temp)}K"),
                    f"{self.base_filename}_{int(prev_temp)}K.traj"
                )
                if os.path.exists(prev_cache):
                    resume_file = prev_cache
                    logger.info(f"Using cache from T={prev_temp}K as resume point")

            # Run MC simulation at current temperature
            self.sampler.run_monte_carlo(
                initial_structure=self.current_structure,
                save_interval=self.save_interval,
                trajectory_file=traj_file,
                resume_file=resume_file,
                log_file=log_file,
                resume=(resume_file is not None),
                annealing = 1
            )

            # Print completion information
            print(f"\nCompleted T = {temp}K simulation")
            print(f"  Saved trajectory: {traj_file}")
            print(f"  Saved log: {log_file}")

        # Print final summary
        print("\n" + "="*80)
        print("Annealing Completed Successfully!")
        print(f"  Final Structure Atoms: {len(self.current_structure)}")
        print(f"  Total Temperatures Processed: {total_current_temps}")
        print("="*80 + "\n")

        return annealing_results
    
    def get_final_structure(self):
        """Get the final structure after annealing"""
        if not hasattr(self, 'current_structure'):
            raise RuntimeError("No structure available. Run annealing first.")
        return self.current_structure.copy()
    
class CanonicalEnsembleSampler(MonteCarloSampler):
    """
    Canonical ensemble sampler specialized for atomic swap operations.
    Fixes swap_ratio=1 to perform only atomic swaps, no element replacements.
    """
    
    def __init__(
        self,
        model,
        analyzer,
        element_list: List[str],
        lattice_constant: float,
        max_steps: int = 1000,
        temperature: float = 1.0
    ):
        # Force swap_ratio=1 to only perform atomic swaps
        super().__init__(
            model=model,
            analyzer=analyzer,
            element_list=element_list,
            lattice_constant=lattice_constant,
            max_steps=max_steps,
            temperature=temperature,
            swap_ratio=1.0
        )
        
    def run_monte_carlo(
        self,
        initial_structure: Atoms,
        save_interval: int = 100,
        trajectory_file: str = 'accepted.traj',
        resume_file = None,
        log_file: str = 'full_log.txt',
        resume: bool = False,
        val1000: bool = False,
        annealing = False
    ) -> List[dict]:
        """
        Execute canonical ensemble Monte Carlo simulation
        Overrides parent method to ensure only swap operations are used
        """
        return super().run_monte_carlo(
            initial_structure=initial_structure,
            save_interval=save_interval,
            trajectory_file=trajectory_file,
            log_file=log_file,
            resume=resume,
            val1000=val1000,
            annealing = annealing
        )