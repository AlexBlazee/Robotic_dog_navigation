#!/bin/bash
#SBATCH --job-name=gpu_job      # Job name
#SBATCH --nodes=1               # Number of nodes
#SBATCH --tasks-per-node=1      # Number of tasks per node
#SBATCH --gres=gpu:4            # Number of GPUs required
#SBATCH --time=20:00:00          # Walltime
#SBATCH --mem=80G                # Memory per node in GB
#SBATCH -G 4
#SBATCH -o logfile

# Load necessary modules or activate environment (if required)
# module load cuda   # Load CUDA module if needed

# Command to execute your GPU program
python3 main_navigation.py