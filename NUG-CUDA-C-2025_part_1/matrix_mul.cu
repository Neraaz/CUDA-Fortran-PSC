#include <cstdio>
#include <cstdlib>

#define N 16

// Kernel function to multiply two matrices
__global__ void matrixMul(int *a, int *b, int *c, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int sum = 0;

    if (col < width && row < width) {
        for (int k = 0; k < width; ++k) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);
    int a[N][N], b[N][N], c[N][N];
    int *d_a, *d_b, *d_c;

    // Initialize host matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = i * N + j;
            b[i][j] = j * N + i;
            c[i][j] = 0;
        }
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 dimBlock(4, 4);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    // Launch kernel
    matrixMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i][j]);
        }
        printf("\\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

