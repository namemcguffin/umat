#!/bin/bash

#SBATCH --time=4:0:0
#SBATCH --mem=400GB
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=5
#SBATCH -o out/slurm/%j.out
#SBATCH -e out/slurm/%j.err

# required env vars:
# SIF_FILE: sif file
# INP_PATH: input data directory path
# OUT_PATH: output path

module load StdEnv/2023 apptainer

set -euxo pipefail
apptainer run \
  -C -B $PWD:/bnd -B $SLURM_TMPDIR:/tmpdir --nv --writable-tmpfs \
  "${SIF_FILE}" \
  bash -c "cd /workdir/ && python segd.py -i '/bnd/${INP_PATH}/images/mosaic_{c}_z{z}.tif' -o '/bnd/${OUT_PATH}' -m cyto3 -tc '-5.5' -s PolyT -n DAPI -d 70 -b 128 -pt /tmpdir -cx 4096 -cy 4096 -cz 7 -z 0 -z 1 -z 2 -z 3 -z 4 -z 5 -z 6"
