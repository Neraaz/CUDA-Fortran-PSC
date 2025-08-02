#!/bin/bash
#SBATCH --job-name=qe-job       # Job name
#SBATCH -N 1 # Number of nodes
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5 # Number of CPU cores per task (adjust as per node config)
##SBATCH --mem=100G                # Total memory requested
#SBATCH -t 00:10:00             # Max time for the job

# Set up the environment
#module load singularity
#cd $SLURM_SUBMIT_DIR

set -x


module load nvhpc

nvprof ./device
