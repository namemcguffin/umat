#!/bin/bash

#SBATCH --time=1:0:0
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=1
#SBATCH -o out/slurm/%j.out
#SBATCH -e out/slurm/%j.err

# required env vars:
# SIF_FILE: sif file
# FTR_DIR: path containing all input segmentation polygon feather files (named in order of z slice)
# DT_FILE: input CSV detected transcripts file
# DTF_FILE: output feather assigned transcripts file
# AD_FILE: output cell/gene matrix matrix file

module load StdEnv/2023 apptainer

INP_FLAGS="$(find "${FTR_DIR}" -type f -exec echo '-i '\''/bnd/{}'\''' \; | sort | tr '\n' ' ' | rg  ' $' -r '')"
set -euxo pipefail
apptainer run \
  -C -B $PWD:/bnd -B $SLURM_TMPDIR:/tmpdir --writable-tmpfs \
  "${SIF_FILE}" \
  bash -c "umat assign ${INP_FLAGS} -d '/bnd/${DT_FILE}' -f '/bnd/${DTF_FILE}' -a '/bnd/${AD_FILE}'"
