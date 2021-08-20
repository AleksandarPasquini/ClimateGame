#!/bin/bash

#SBATCH --job-name="bulid_results_graphs"
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16384
#SBATCH --time=0-1:00:00

PROJECTDIR=${HOME}

echo -n "started at: "; date

#module load R/3.2.1-vlsci_intel-2015.08.25
module load R/3.2.1-GCC-4.9.2

time collect_results.sh

time Rscript plotCooperatorsEndHeatmap.R results.csv end_sim_heatmap

times

echo -n "ended at: "; date
