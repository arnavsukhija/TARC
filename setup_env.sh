#!/bin/bash

# Ensure we are running a login-like shell environment so `module` and `conda` commands exist
if [ -f /etc/profile ]; then
    . /etc/profile
fi

# Load necessary modules for ETH Euler (adjust based on compiler/CUDA version available on Euler)
# This example uses a typical recent CUDA combination. Check `module avail cuda` on Euler for exact versions.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

module purge || true

# In modern Euler, the easiest way to avoid LMOD conflicts ("module spider gcc" errors)
# is to let JAX download its own bundled CUDA libraries (cuSPARSE, cuDNN, etc.) via pip.
# We will just load the base Python/Conda and rely on the pip `nvidia-*` packages.

# Optionally load a basic cuDNN if your cluster defaults to it (as shown by your `module list`).
# module load cudnn/9.8.0 

# We do NOT set CUDA_HOME manual LD_LIBRARY_PATHs anymore because pip will supply them.

# Set up Conda or virtual environment.
# Assuming you use miniconda for your environment named "tarc_jax":
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
else
    export PATH="$CONDA_BASE/bin:$PATH"
fi
conda activate tarc_jax

# Add the root TARC directory to PYTHONPATH so imports like `optimizer` work from anywhere
export PYTHONPATH="$HOME/TARC:$PYTHONPATH"

# Alternatively, if you use a standard python venv:
# source /cluster/home/asukhija/tarc_venv/bin/activate

# ==============================================================================
# ONE-TIME INSTALLATION INSTRUCTIONS (Run this manually once after sourcing this script):
# To fix the GPU issue with JAX on the cluster, you MUST install the CUDA-enabled JAX wheel:
# pip install -U "jax[cuda12]"
# ==============================================================================

# Ensure CUDA runtime logic is exposed if needed by custom ops.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
