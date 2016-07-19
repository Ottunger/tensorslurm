#!/bin/bash
#SBATCH --job-name=Model
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=6
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32 # megabytes
#SBATCH --partition=PartitionName
#
#SBATCH --comment=MyModel

srun --multi-prog slurm.conf