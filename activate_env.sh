#!/usr/bin/env bash
module purge
module load mamba
source deactivate
source activate ligpar
module load gromacs-2023.3-openmpi-cuda-qx