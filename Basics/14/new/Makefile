FC = nvfortran
FCFLAGS = -cuda -gpu=cc70,cuda12.6,rdc -c
CC = nvcc
CFLAGS = -c -arch=sm_70

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = /ocean/projects/pscstaff/nnepal/nvc_hpc/Linux_x86_64/25.1/cuda/lib64
LIBS = -cuda -gpu=cc70,cuda12.6,rdc -c++libs

OBJS = prog1.o mod.o
exe  = f90out

all: $(exe)

$(exe): $(OBJS)
	$(FC) -o $(exe) $(OBJS) -L$(DIR_LIB) $(LIBS)

prog1.o: prog1.f90
	$(FC) $(FCFLAGS) prog1.f90
mod.o: mod.cu
	$(CC) $(CFLAGS) mod.cu

clean:
	rm -rf *.o *.mod $(exe)

