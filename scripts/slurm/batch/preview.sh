#!/bin/bash

#SBATCH --time=2:0:0
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=1
#SBATCH -o out/slurm/%j.out
#SBATCH -e out/slurm/%j.err

# required env vars:
# SIF_FILE: sif file
# INP_PATH: input data directory path
# NPY_PATH: input segmentation npy file path
# Z_SLICE: z-slice to generate preview for
# OUT_PATH: output preview image path

module load StdEnv/2023 apptainer

set -euxo pipefail
apptainer run \
  -C -B $PWD:/bnd -B $SLURM_TMPDIR:/tmpdir --writable-tmpfs \
  "${SIF_FILE}" \
  bash -c "umat preview -i '/bnd/${INP_PATH}/images/mosaic_{c}_z{z}.tif' -c 'PolyT' -n 'DAPI' -z '${Z_SLICE}' -m '/bnd/${NPY_PATH}' -o '/bnd/${OUT_PATH}'"
