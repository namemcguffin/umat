#!/bin/bash

#SBATCH --time=24:0:0
#SBATCH --mem=490GB
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=5
#SBATCH -o out/slurm/%j.out
#SBATCH -e out/slurm/%j.err

# required env vars:
# SIF_FILE: sif file
# INP_PATH: input data directory path
# MD_PATH: custom model path
# OUT_PATH: output path

module load StdEnv/2023 apptainer

set -euxo pipefail
apptainer run \
  -C -B $PWD:/bnd -B $SLURM_TMPDIR:/tmpdir --nv --writable-tmpfs \
  "${SIF_FILE}" \
  bash -c "umat segd -i '/bnd/${INP_PATH}/images/mosaic_{c}_z{z}.tif' -o '/bnd/${OUT_PATH}' -w '/bnd/${MD_PATH}' -c PolyT -n DAPI -b 128 -pt /tmpdir -lx 4096 -ly 4096 -lz 7 -z 0 -z 1 -z 2 -z 3 -z 4 -z 5 -z 6 -ts 0.25"
