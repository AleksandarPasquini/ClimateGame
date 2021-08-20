#!/bin/bash

#SBATCH --job-name="SH_N_one"
#SBATCH --partition=physical
#SBATCH --ntasks=5
#SBATCH --time=0-500:00:00

echo -n "started at: "; date

module load OpenMPI/1.10.2-GCC-4.9.2
module load Python/2.7.9-GCC-4.9.2

time mpirun python runNPlayers.py

times
echo -n "ended at: "; date

