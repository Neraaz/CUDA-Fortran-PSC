# CUDA C++ Examples

This repository contains four introductory CUDA C++ programs that demonstrate basic concepts of CUDA programming. These examples are designed to be run on Perlmutter at NERSC and cover various aspects of GPU programming, including memory management, kernel launches, and parallel processing using threads and blocks.

## Overview

1. **Hello World**
2. **Vector Addition**
3. **Matrix Multiplication**
4. **Dot Product**

## Programs

### 1. Hello World

This example introduces the basic structure of a CUDA program with a simple kernel that prints "Hello World!" from the device.

**File:** `hello_world.cu`

**Kernel Function:**
```cpp
__global__ void helloFromGPU() {
    printf("Hello, World! from thread %d\n", threadIdx.x);
}
### 2. Vector Addition
This example demonstrates a basic vector addition using CUDA, where each thread computes one element of the resulting vector.

File: vector_add.cu

Kernel Function:
