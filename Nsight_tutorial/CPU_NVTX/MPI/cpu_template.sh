#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH -N 1
#SBATCH -p RM
#SBATCH --ntasks-per-node=NOMP
#SBATCH -t 12:00:00

set -x
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5

#export OMP_NUM_THREADS=NOMP
mpirun -np NOMP nsys profile -t nvtx,mpi -o NSYS_OUT EXECUTABLE
