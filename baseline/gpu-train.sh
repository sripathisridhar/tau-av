#!/bin/bash -l
#SBATCH --job-name=tau-baseline
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Purge any module loaded by default
module purge > /dev/null 2>&1

conda activate tau-torch
srun python main.py