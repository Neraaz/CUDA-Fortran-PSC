# Makefile for CUDA C++ examples on Perlmutter at NERSC

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_80

# Targets
TARGETS = hello_world vector_add matrix_mul dot_product

# Rules to build all targets
all: $(TARGETS)

# Rule to build hello_world
hello_world: hello_world.cu
	$(NVCC) $(NVCC_FLAGS) -o hello_world hello_world.cu

# Rule to build vector_add
vector_add: vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o vector_add vector_add.cu

# Rule to build matrix_mul
matrix_mul: matrix_mul.cu
	$(NVCC) $(NVCC_FLAGS) -o matrix_mul matrix_mul.cu

# Rule to build dot_product
dot_product: dot_product.cu
	$(NVCC) $(NVCC_FLAGS) -o dot_product dot_product.cu

# Clean rule
clean:
	rm -f $(TARGETS)
	rm -f *.o

