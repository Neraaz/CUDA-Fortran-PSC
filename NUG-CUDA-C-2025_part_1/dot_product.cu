#include <cstdio>
#include <cstdlib>

__global__ void dotProduct(int *a, int *b, int *c, int n) {
    __shared__ int temp[256];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    temp[threadIdx.x] = (index < n) ? a[index] * b[index] : 0;

    __syncthreads();

    if (0 == threadIdx.x) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp[i];
        }
        atomicAdd(c, sum);
    }
}

int main() {
    int n = 1000;
    int size = n * sizeof(int);
    
    // Allocate memory on the host
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(sizeof(int));

    // Initialize vectors
    for(int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }
    *h_c = 0;

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, sizeof(int));

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    dotProduct<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("Dot Product: %d\\n", *h_c);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

