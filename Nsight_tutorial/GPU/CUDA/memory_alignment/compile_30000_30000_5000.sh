#!/bin/bash
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5
export HPC_SDK="/opt/packages/nvhpc/v25.5/Linux_x86_64/25.5"

nvcc -O3 -g -arch=sm_90 \
  -lineinfo \
    -o heat2d_30000_30000_5000 heat2d_gpu.cu
