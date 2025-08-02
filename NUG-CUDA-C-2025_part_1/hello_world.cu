#include <cstdio>

// Kernel function to print from the device
__global__ void helloFromGPU() {
    printf("Hello, World! from thread %d\n", threadIdx.x);
}

int main() {
    // Launch kernel with 1 block of 10 threads
    helloFromGPU<<<1, 10>>>();
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    return 0;
}

