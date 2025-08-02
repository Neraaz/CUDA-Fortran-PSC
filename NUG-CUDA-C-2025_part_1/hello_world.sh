#!/bin/bash
#SBATCH -N 1
#SBATCH -G 2 
#SBATCH -t 5
#SBATCH -C gpu 
#SBATCH -A trn017 #for this training series for part 1 and part 2
#SBATCH -q shared
##SBATCH --reservation=nug_cuda_c #this line only during training until 2pm
./hello_world

