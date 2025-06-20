# UMAT: untitled merscope analysis tool

## notice

this softare is still in an experimental/unstable state, and further testing is required before it can be recommended for general use, as bugs might still be present.
if you are interested in using it and/or encounter any issues, please feel free to file issues on github and/or reach out through other channels.

## description

this repository provides a set of scripts which allow for segmentation, boundary determination, and transcript assignment following MERSCOPE imaging.

all scripts provide a `-h` flag which describe necessary arguments, and are located in the `src` subdirectory.

## installation and use

`uv` is used for package management.

dependencies can be installed in virtualenv via the following commands:
```bash
uv venv # create venv
uv pip install -r requirements.txt # install deps
```

a `Dockerfile` is also provided, which can be used to generate a OCI image and subsequently a SIF file for use on HPC clusters via the following commands:
```bash
docker build --platform=linux/amd64 -t umat:latest . # build docker image
docker save umat:latest > umat.tar # save docker image to disk as archive
apptainer build umat.sif docker-archive://umat.tar # convert docker archive to sif
```
the image generated will have the `cpsam` cellpose model pre-downloaded.
if you plan on using other cellpose models in an HPC environment where compute nodes do not have an internet connection, you should change the Dockerfile accordingly to ensure that the models are cached in the image.

## segmentation script

`segd.py` provides a way to segment MERSCOPE-generated mosaic files using cellpose, specifically utilizing the `distributed_eval` function provided in `cellpose.contrib` to split work into chunks for distribution across multiple workers.

the output label array is saved to disk in the npy format (ingestible via `numpy.load`).

it is recommended to run `segd.py` on HPC infrastructure as it is extremely compute and memory intensive.
only linux x86-64 environments are supported for segmentation, and the presence of a CUDA-compatible GPU is assumed.

`segd.py` is able to utilize multiple GPUs simultaneously, with recommended allocation being N+1 CPUs allocated with N GPUs.

### `segd.py` chunk parameters

the chunk side length (`lx`, `ly`, `lz`) parameters should be optimized for depending on node specifics to maximally utilize available memory.

on a 400GB RAM, 5 CPU, 4 A100 node allocation, chunk dimension was set to (4096, 4096, 7) to avoid OOM-death while using close to maximal available resources.

## post-segmentation scripts

these scripts are much less compute and memory intensive than the segmentation step, generally requiring only one CPU and ~20GBs of RAM (i.e. they can be run on personal/work computers).

`preview.py` provides a way to generate a preview of the segmentations generated by `segd.py`.

`boundary.py` generates cell boundary polygons using the masks generated by `segd.py`, saving it as a geopandas-generated feather file.
these can be read in using `geopandas.read_feather` in python and `sfarrow::st_read_feather` in R.

`assign.py` generates a cell by gene matrix using the cell boundary polygons generated by `boundary.py`, saving it as an anndata h5ad file.
for very large datasets, `assign.py` might use a lot of RAM, as such running it on HPC resources might be advisable.

## re-training scripts

`sample.py` is used to generate a HDF5 file of random sub-selections of a provided image, useful for creating a training dataset.
each sample is represented as a top-level dataset in the resulting HDF5 file.

`addlab.py` takes an ImageJ generated ROI file and adds it to a specified sample HDF5 group from a `sample.py`-generated file.

`retrain.py` uses a `sample.py`-generated file with added labels to fine-tune an existing cellpose model.

## segmentation-free data generation

`spot.py` generates cell by gene matrix without using any prior cell segmentation, instead binning all transcripts into "pseudo-spots"

## provided SLURM scripts

to facilitate use of the segmentation pipeline on HPC infrastructure (assuming SLURM use for scheduling) a set of scripts are provided under the `scripts/slurm` subdirectory, providing a complete segmentation pipeline.
the recommended manner of use is to invoke one of the scripts under `scripts/slurm/run`, which will use the scripts under `scripts/slurm/batch` to set up a series of SLURM jobs to run cell segmentation.
