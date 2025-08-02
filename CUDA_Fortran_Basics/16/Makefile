FC = nvfortran
FCFLAGS = -cuda -c

WHICH_FC = $(shell which $(FC))
DIR_FC = $(shell dirname $(WHICH_FC))
DIR_LIB = /ocean/projects/pscstaff/nnepal/nvc_hpc/Linux_x86_64/25.1/compilers/lib
LIBS =-cuda -mp -cudalib=cublas -gpu=cc70,cuda12.6 -lblas

OBJS = prog1.o
exe  = f90out

all: $(exe)

$(exe): $(OBJS)
	$(FC) -o $(exe) $(OBJS) -L$(DIR_LIB) $(LIBS)

prog1.o: prog1.f90
	$(FC) $(FCFLAGS) prog1.f90

clean:
	rm -rf *.o *.mod $(exe)

