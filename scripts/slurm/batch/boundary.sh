#!/bin/bash

#SBATCH --time=2:0:0
#SBATCH --mem=320GB
#SBATCH --cpus-per-task=1
#SBATCH -o out/slurm/%j.out
#SBATCH -e out/slurm/%j.err

# required env vars:
# SIF_FILE: sif file
# NPY_PATH: input segmentation npy file path
# OUT_PATH: output feather file path
# MP_PATH: input micron to pixel file path
# Z_SLICE: z slice to determine boundaries for

module load StdEnv/2023 apptainer

set -euxo pipefail
apptainer run \
  -C -B $PWD:/bnd -B $SLURM_TMPDIR:/tmpdir --writable-tmpfs \
  "${SIF_FILE}" \
  bash -c "umat boundary -i '/bnd/${NPY_PATH}' -o '/bnd/${OUT_PATH}' -m '/bnd/${MP_PATH}' -z ${Z_SLICE} -j ${SLURM_CPUS_ON_NODE}"
