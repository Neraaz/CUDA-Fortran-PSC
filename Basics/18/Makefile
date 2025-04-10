FC = nvfortran
FCFLAGS = -cuda -c

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = /ocean/projects/pscstaff/nnepal/nvc_hpc/Linux_x86_64/25.1/compilers/lib
LIBS =-cuda -mp -cudalib=cublas -gpu=cc70,cuda12.6,lineinfo -lblas

OBJS = prog2.o
exe  = f90out2

all: $(exe)

$(exe): $(OBJS)
	$(FC) -o $(exe) $(OBJS) -L$(DIR_LIB) $(LIBS)

prog2.o: prog2.f90
	$(FC) $(FCFLAGS) prog2.f90

clean:
	rm -rf *.o *.mod $(exe)

