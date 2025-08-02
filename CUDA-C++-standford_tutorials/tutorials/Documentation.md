1. I am using NVHPC 25.1 version which uses cuda 12.8.
nvcc --version
2. But "nvidia-smi" shows cuda 12.6 version.

It seems to issue error: CUDA error: the provided PTX was compiled with an unsupported toolchain,
with following command:
nvcc test.cu -o test

This is due to the cuda version mismatch. Potential solutions are:

a. (Recommended) Compile SASS (Streaming Assembler, GPU machine code) only code (especially configuring for H200, won't be portable to other machines) by specifying architecture:

nvcc -arch=sm_90 test.cu -o test

b. Compile SASS + PTX

nvcc -gencode=arch=compute_90,code=sm_90 \
     -gencode=arch=compute_90,code=compute_90 \
     -o test test.cu

c. Match nvidia-smi and nvcc driver version to either 12.6 or 12.8.




