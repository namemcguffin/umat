#!/bin/bash

set -euxo pipefail
shopt -s nullglob

ACCOUNT="${1}"
SIF_FILE="${2}"
INP_DIR="${3}"
NPY_PATH="${4}"
FTR_DIR="${5}"
DTF_PATH="${6}"
AD_PATH="${7}"
IMG_DIR="${8}"

mkdir -p "$(dirname "${NPY_PATH}")"
mkdir -p "${FTR_DIR}"
mkdir -p "$(dirname "${DTF_PATH}")"
mkdir -p "$(dirname "${AD_PATH}")"
mkdir -p "${IMG_DIR}"

SEG_ID="$(sbatch --parsable --account="${ACCOUNT}" \
  --export="SIF_FILE=${SIF_FILE},INP_PATH=${INP_DIR},OUT_PATH=${NPY_PATH}" \
  batch/segd.sh \
)"

BND_DEP_STR='afterok'
for z in $(seq 0 6); do
  BND_DEP_STR+=":$(sbatch --parsable --account="${ACCOUNT}" \
    -d "afterok:${SEG_ID}" \
    --export="SIF_FILE=${SIF_FILE},NPY_PATH=${NPY_PATH},OUT_PATH=${FTR_DIR}/z${z}.feather,MP_PATH=${INP_DIR}/images/micron_to_mosaic_pixel_transform.csv,Z_SLICE=${z}" \
    batch/boundary.sh \
  )"
done

sbatch --parsable --account="${ACCOUNT}" \
  -d "${BND_DEP_STR}" \
  --export="SIF_FILE=${SIF_FILE},FTR_DIR=${FTR_DIR},DT_FILE=$(find "${INP_DIR}" -maxdepth 1 -name '*detected_transcripts*'),DTF_FILE=${DTF_PATH},AD_FILE=${AD_PATH}" \
  batch/assign.sh

for z in $(seq 0 6); do
  sbatch --parsable --account="${ACCOUNT}" \
    -d "afterok:${SEG_ID}" \
    --export="SIF_FILE=${SIF_FILE},INP_PATH=${INP_DIR},NPY_PATH=${NPY_PATH},Z_SLICE=${z},OUT_PATH=${IMG_DIR}/z${z}.png" \
    batch/preview.sh
done
