#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>

extern "C" void cuda_zgemm_c_(int* m, int* n, int* k,
                             const cuDoubleComplex* alpha,
                             const cuDoubleComplex* A, int* lda,
                             const cuDoubleComplex* B, int* ldb,
                             const cuDoubleComplex* beta,
                             cuDoubleComplex* C, int* ldc)
{
    // Set device to prefer unified memory
    cudaSetDevice(0);
    cudaDeviceSetMemFlags(cudaMemAttachGlobal);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Determine cuBLAS operation modes
    cublasOperation_t cuTransA = CUBLAS_OP_N;
    cublasOperation_t cuTransB = CUBLAS_OP_N;

    // Allocate unified memory that's accessible from both CPU and GPU
    size_t size_A = (*lda) * (*k);
    size_t size_B = (*ldb) * (*n);
    size_t size_C = (*ldc) * (*n);
    
    printf("M = %d, N = %d, K = %d\n", *m, *n, *k);
    printf("LDA = %d, LDB = %d, LDC = %d\n", *lda, *ldb, *ldc);

    // Allocate unified memory
    cuDoubleComplex *u_A, *u_B, *u_C;
    cudaMallocManaged(&u_A, sizeof(cuDoubleComplex) * size_A);
    cudaMallocManaged(&u_B, sizeof(cuDoubleComplex) * size_B);
    cudaMallocManaged(&u_C, sizeof(cuDoubleComplex) * size_C);

    // Copy data to unified memory (no need for explicit device copies)
    memcpy(u_A, A, sizeof(cuDoubleComplex) * size_A);
    memcpy(u_B, B, sizeof(cuDoubleComplex) * size_B);
    
    // Prefetch data to GPU for better performance
    cudaMemPrefetchAsync(u_A, sizeof(cuDoubleComplex) * size_A, 0);
    cudaMemPrefetchAsync(u_B, sizeof(cuDoubleComplex) * size_B, 0);
    cudaMemPrefetchAsync(u_C, sizeof(cuDoubleComplex) * size_C, 0);

    // Perform ZGEMM operation
    cublasStatus_t status = cublasZgemm(handle, cuTransA, cuTransB,
                                      *m, *n, *k,
                                      alpha,
                                      u_A, *lda,
                                      u_B, *ldb,
                                      beta,
                                      u_C, *ldc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasZgemm failed with error: " << status << std::endl;
    }

    // Wait for computation to finish
    cudaDeviceSynchronize();

    // Copy result back (not strictly needed with unified memory, but maintains interface)
    memcpy(C, u_C, sizeof(cuDoubleComplex) * size_C);

    // Free unified memory and destroy handle
    cudaFree(u_A);
    cudaFree(u_B);
    cudaFree(u_C);
    cublasDestroy(handle);
}
