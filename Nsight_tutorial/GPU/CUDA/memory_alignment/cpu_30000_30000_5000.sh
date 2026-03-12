#!/bin/bash
#SBATCH --job-name=heat-30000_30000_5000
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH -t 6:00:00

set -x
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5

#nsys profile -o nsys_30000_30000_5000 ./heat2d_30000_30000_5000
ncu --launch-count 2 -o ncu_30000_30000_5000 ./heat2d_30000_30000_5000
