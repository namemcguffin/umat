from dataclasses import dataclass
from functools import wraps
from gc import collect
from inspect import signature
from os import mkdir
from os.path import basename
from pathlib import Path
from typing import Annotated

import cappa
import numpy as np
from cellpose.contrib import distributed_segmentation
from cellpose.io import logger_setup
from cellpose.models import CellposeModel
from dask_cuda.local_cuda_cluster import LocalCUDACluster
from distributed import Client
from tifffile import imread
from zarr import Array as ZArray
from zarr import array as zarray

# below needed since get_block_crops assumes that overlap is always
# an int but it will be a float if diameter is set to a float
_gbc = distributed_segmentation.get_block_crops


@wraps(_gbc)
def wgbc(*args, **kwargs):
    if "overlap" in kwargs:
        kwargs["overlap"] = round(kwargs["overlap"])
    else:
        args = list(args)
        args[2] = round(args[2])
    return _gbc(*args, **kwargs)


distributed_segmentation.get_block_crops = wgbc


@dataclass
class DistributedSegConf:
    img_fmt: Annotated[
        str,
        cappa.Arg(
            short="-i",
            help=(
                "pattern for input mosaic files, in python format string format."
                " following patterns assumed present: 'c' (for channel) and 'z' (for z stack level)."
                " example: 'data_dir/region_0/images/mosaic_{c}_z{z}.tif'."
            ),
        ),
    ]
    cyt_pat: Annotated[str, cappa.Arg(short="-c", help="name of cytoplasm channel. example: 'PolyT'.")]
    nuc_pat: Annotated[str, cappa.Arg(short="-n", help="name of nuclear channel. example: 'DAPI'.")]
    z_slices: Annotated[
        list[int],
        cappa.Arg(
            short="-z",
            action=cappa.ArgAction("append"),
            help="z slices to consider for segmentation. can be provided multiple times to specify multiple slices.",
        ),
    ]
    out_path: Annotated[Path, cappa.Arg(short="-o", help="path for output masks npy file")]
    tempdir: Annotated[Path, cappa.Arg(short="-pt", help="path to temporary directory")]
    diameter: Annotated[
        int | None,
        cappa.Arg(
            short="-d",
            help="cell diameter in pixels provided to cellpose. if left unset will be determined autonomously by cellpose.",
        ),
    ] = None
    batch_size: Annotated[int, cappa.Arg(short="-b", help="batch size specified to 'Cellpose.eval'")] = (
        signature(CellposeModel.eval).parameters["batch_size"].default
    )
    model_path: Annotated[
        Path | None,
        cappa.Arg(short="-w", help="optional path to cellpose model weights to use (uses cpsam if unset)"),
    ] = None
    chunk_x: Annotated[int, cappa.Arg(short="-lx", help="chunk block x-axis length for distributed processing")] = 256
    chunk_y: Annotated[int, cappa.Arg(short="-ly", help="chunk block y-axis length for distributed processing")] = 256
    chunk_z: Annotated[int, cappa.Arg(short="-lz", help="chunk block z-axis length for distributed processing")] = 256
    cellprob_threshold: Annotated[float, cappa.Arg(short="-tc", help="cellprob_threshold specified to 'Cellpose.eval'")] = (
        signature(CellposeModel.eval).parameters["cellprob_threshold"].default
    )
    flow_threshold: Annotated[
        float | None,
        cappa.Arg(
            short="-tf", help="optional flow_threshold specified to 'Cellpose.eval', ignored if stitch_threshold is left unset"
        ),
    ] = None
    stitch_threshold: Annotated[
        float | None,
        cappa.Arg(
            short="-ts",
            help="optional stitch_threshold specified to 'Cellpose.eval', leave unset to use 'true' 3D segmentation",
        ),
    ] = None


def run_segd(conf: DistributedSegConf, cluster: LocalCUDACluster):
    cyt_paths = [Path(conf.img_fmt.format(c=conf.cyt_pat, z=z)) for z in conf.z_slices]
    print(f"creating cytoplasm channel zarr array from paths {cyt_paths}", flush=True)
    cyt_zarr = zarray(
        np.stack([imread(p, aszarr=False) for p in cyt_paths], axis=0),
        store=conf.tempdir / "seg.zarr",
        chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
    )

    nuc_paths = [Path(conf.img_fmt.format(c=conf.nuc_pat, z=z)) for z in conf.z_slices]
    print(f"creating nuclear channel zarr array from paths {nuc_paths}", flush=True)
    nuc_zarr = zarray(
        np.stack([imread(p, aszarr=False) for p in nuc_paths], axis=0),
        store=conf.tempdir / "nuc.zarr",
        chunks=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
    )

    mkdir(conf.tempdir / "cellpose_temp")

    print("running distributed_eval", flush=True)
    masks, _ = distributed_segmentation.distributed_eval(
        input_zarr=cyt_zarr,
        blocksize=(conf.chunk_z, conf.chunk_x, conf.chunk_y),
        write_path=str(conf.tempdir / "out.zarr"),
        # insert nuclear channel during preprocessing step as documented
        # here: https://cellpose.readthedocs.io/en/latest/distributed.html
        preprocessing_steps=[
            (
                lambda image, crop: np.stack(
                    (
                        image,
                        nuc_zarr[crop],  # noqa: F821 - ruff seems to not handle lambda capture properly (?)
                        image * 0,
                    ),
                    axis=-1,
                ),
                {},
            )
        ],
        model_kwargs={
            "gpu": True,
        }
        | ({"pretrained_model": str(conf.model_path)} if conf.model_path is not None else {}),
        eval_kwargs={
            "batch_size": conf.batch_size,
            "channel_axis": None if conf.nuc_pat is None else -1,
            "z_axis": 0,
            "cellprob_threshold": conf.cellprob_threshold,
        }
        | ({"diameter": conf.diameter} if conf.diameter is not None else {})
        | (
            {
                "do_3D": True,
            }
            if conf.stitch_threshold is None
            else {
                "stitch_threshold": conf.stitch_threshold,
                "flow_threshold": conf.flow_threshold,
            }
        ),
        cluster=cluster,
        temporary_directory=str(conf.tempdir / "cellpose_temp"),
    )

    # sanity check to make sure masks is of right type
    assert isinstance(masks, ZArray), f"expected masks to be zarr.Array, got {type(masks)}"

    # attempt to clear up memory space
    # before loading masks into memory
    del cyt_zarr
    del cyt_paths
    del nuc_zarr
    del nuc_paths
    collect()

    with open(conf.out_path, "wb") as nf:
        np.save(
            nf,
            masks[:],
        )


def run():

    conf = cappa.parse(cappa.Command(DistributedSegConf, name=basename(__file__)), completion=False)
    print(f"config: {conf}", flush=True)

    logger_setup()

    # NOTE: not using context manager due to crashes during __exit__ call

    # monkey-patch necessary since distributed_eval
    # assumes presence of a .client field in cluster
    cluster = LocalCUDACluster(
        local_directory=str(conf.tempdir / "dask_cuda_tmp"),
        shared_filesystem=True,
    )
    cluster.client = Client(cluster)  # pyright: ignore

    run_segd(conf, cluster)

    # don't fail from timeout errors on client/cluster close
    try:
        cluster.close()
    except TimeoutError:
        print("timeout error on cluster close", flush=True)
    try:
        cluster.client.close()  # pyright: ignore
    except TimeoutError:
        print("timeout error on client close", flush=True)


if __name__ == "__main__":
    run()
