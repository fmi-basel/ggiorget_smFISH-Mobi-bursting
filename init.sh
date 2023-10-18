#!/bin/bash
export MAMBA_EXE="$(pwd)/infrastructure/apps/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="$(pwd)/infrastructure/apps/micromamba"
export MAMBA_ROOT_ENVIRONMENT="$(pwd)/infrastructure/apps/micromamba"

eval "$($MAMBA_ROOT_ENVIRONMENT/bin/micromamba shell hook -s posix)"

export PIP_CACHE_DIR="$(pwd)/infrastructure/apps/micromamba/PIP_CACHE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
