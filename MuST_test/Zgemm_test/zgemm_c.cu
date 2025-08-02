// cuda_zgemm.cu
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

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Determine cuBLAS operation modes
    cublasOperation_t cuTransA = CUBLAS_OP_N;
    cublasOperation_t cuTransB = CUBLAS_OP_N;

    // Allocate device memory
    cuDoubleComplex *d_A, *d_B, *d_C;
    size_t size_A = (*lda) * (*k);
    size_t size_B = (*ldb) * (*n);
    size_t size_C = (*ldc) * (*n);
    printf("M = %d, N = %d, K = %d\n", *m, *n, *k);
    printf("LDA = %d, LDB = %d, LDC = %d\n", *lda, *ldb, *ldc);

    cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * size_A);
    cudaMalloc((void**)&d_B, sizeof(cuDoubleComplex) * size_B);
    cudaMalloc((void**)&d_C, sizeof(cuDoubleComplex) * size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeof(cuDoubleComplex) * size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(cuDoubleComplex) * size_B, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_C, C, sizeof(cuDoubleComplex) * size_C, cudaMemcpyHostToDevice);

    // Perform ZGEMM operation
    cublasStatus_t status = cublasZgemm(handle, cuTransA, cuTransB,
                                      *m, *n, *k,
                                      alpha,
                                      d_A, *lda,
                                      d_B, *ldb,
                                      beta,
                                      d_C, *ldc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasZgemm failed with error: " << status << std::endl;
    }

    // Copy result back to host
    cudaMemcpy(C, d_C, sizeof(cuDoubleComplex) * size_C, cudaMemcpyDeviceToHost);

    // Free device memory and destroy handle
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}
