#!/bin/bash
mkdir micromamba
cd micromamba

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

export MAMBA_EXE="$(pwd)/bin/micromamba"
export MAMBA_ROOT_PREFIX="$(pwd)/micromamba"
export MAMBA_ROOT_ENVIRONMENT="$(pwd)/micromamba"

eval "$($(pwd)/bin/micromamba shell hook -s posix)"

mkdir PIP_CACHE

cd ..

export PIP_CACHE_DIR="$(pwd)/micromamba/PIP_CACHE"

micromamba config append channels conda-forge

micromamba config list
