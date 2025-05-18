module load nvhpc

nvfortran -cuda mod.f90 prog.f90 -o device


If there is Makefile,
just use `make` command.

Otherwise compile with nvfortran and match output inside run.sh
