#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 6:00:00

set -x
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5

ncu --launch-count 2 --section SpeedOfLight_RooflineChart -o report_original EXECUTABLE
