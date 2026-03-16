#!/bin/bash

# 1. Initialize ETH Euler Environment
# This is required for 'module' and 'conda' to function correctly in non-interactive Slurm jobs.
if [ -f /cluster/apps/local/etc/setup.sh ]; then
    source /cluster/apps/local/etc/setup.sh
elif [ -f /etc/profile ]; then
    source /etc/profile
fi

# 2. Setup Conda
# Ensure the conda command is available
export CONDA_EXE="$HOME/miniconda3/bin/conda"
export CONDA_PYTHON_EXE="$HOME/miniconda3/bin/python"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate tarc_jax
else
    # Fallback to direct path
    export PATH="$HOME/miniconda3/bin:$PATH"
    source activate tarc_jax
fi

# 3. Environment Variables for JAX/MuJoCo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="egl"
export PYTHONPATH="$HOME/TARC:$PYTHONPATH"

# Redirect Menagerie to Scratch (where quota is high)
export MUJOCO_MENAGERIE_PATH="/cluster/scratch/asukhija/mujoco_menagerie"

# XLA Flags
if [ ! -z "$CUDA_HOME" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi
