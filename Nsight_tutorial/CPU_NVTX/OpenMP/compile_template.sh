#!/bin/bash
module use -a /opt/modulefiles/preproduction
module load nvhpc/25.5
export HPC_SDK="/opt/packages/nvhpc/v25.5/Linux_x86_64/25.5"

nvc -O3 -mp -Minfo=mp,vect -I$HPC_SDK/cuda/include/nvtx3 \
    -L$HPC_SDK/cuda/lib64 -lnvtx3interop \
    -o EXECUTABLE heat2d_cpu.c
