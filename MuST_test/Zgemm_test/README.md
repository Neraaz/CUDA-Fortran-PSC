Run ``./interact.sh`` to get interactive session on Bridges2 H100 1 node 1 gpu for 30 min

##Load modules
module use /opt/packages/nvhpc/v25.5/modulefiles/
ml nvhpc/25.5

nvcc --version ==> CUDA toolkit chain compiler
nvc, nvc++, nvfortran ==> NVIDIA HPC SDK
## Update Makefile based on GPU compute capability (cc90 for H100) and cuda toolkit chain version (12.9)

Run ``make`` for CPU compilation and run ``./text_exe``

Run ``make GPU=1`` for GPU compilation and run ``./text_exe``
