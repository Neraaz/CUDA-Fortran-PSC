# Makefile for CUDA or CPU builds of test_zgemm

# Usage:
#   make          # CPU version
#   make GPU=1    # GPU version with cuBLAS
#   make clean    # Clean build files

# === Compiler and flags ===
FC    := nvfortran
NVCC  := nvcc
F90FLAGS := 
LDLIBS_CPU := -lblas
LDLIBS_GPU := -cuda -O3 -gpu=cc90,cuda12.9,lineinfo -cudalib=cublas -lstdc++
TARGET := test_exec

# === Sources and objects ===
F90SRC := test.F90 zgemm_accel.F90
CU_SRC := zgemm_c.cu
F90OBJ := test.o zgemm_accel.o
CU_OBJ := zgemm_c.o

ifeq ($(GPU),1)
    DEFS := -DCUDA
    LDLIBS := $(LDLIBS_GPU)
    OBJS := $(F90OBJ) $(CU_OBJ)
else
    DEFS :=
    LDLIBS := $(LDLIBS_CPU)
    OBJS := $(F90OBJ)
endif

# === Default rule ===
all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(OBJS) -o $@ $(LDLIBS)

%.o: %.F90
	$(FC) -c $< -o $@ $(DEFS)

%.o: %.f90
	$(FC) -c $< -o $@ $(DEFS)

%.o: %.cu
	$(NVCC) -c $< -o $@

# === Clean rule ===
clean:
	rm -f *.o *.mod $(TARGET)
