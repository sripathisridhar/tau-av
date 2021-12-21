#!/bin/bash -l
#SBATCH --job-name=av-eval
#SBATCH --output=%x.%j.out # %x.%j expands to JobName.JobID
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=datasci
#SBATCH --gres=gpu:1
#SBATCH --mem=4G

# Purge any module loaded by default
module purge > /dev/null 2>&1

conda activate tau-torch
srun python evaluate.py --features_path /research/mc232/tau-av/features_data \
--model_type audio_video