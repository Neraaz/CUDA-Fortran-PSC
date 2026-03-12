#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -t 1:00:00

set -x
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5

export OMP_NUM_THREADS=NOMP

nsys profile -o NSYS_OUT EXECUTABLE
