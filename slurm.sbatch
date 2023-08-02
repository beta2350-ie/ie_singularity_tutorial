#!/bin/bash
#SBATCH --job-name fcn_resnet101
#SBATCH --output logs/%x-%j.log
#SBATCH --error logs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --gpus tesla:1

date
singularity exec --nv ~/workspace/singularity_tutorial/pytorch.sif python fcn_resnet101.py
date
